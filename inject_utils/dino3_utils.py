import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import deepspeed

pretrain='/e2e-data/evad-tech-vla/wangjie68/experiments-VLA/dinov3/pretrain/dinov3-vith16plus-pretrain-lvd1689m'

    
def preprocess_dino_images(example, resize=(224, 224)):
    processor = AutoImageProcessor.from_pretrained(pretrain)

    img_path_list = []
    for exp in example["messages"]:
        if exp.get("role") == "user":
            content_list = exp.get("content", [])

    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        resized_height, resized_width = content["resized_height"], content["resized_width"]
        img_path = image_path.replace("file://", "")
        img_path_list.append(img_path)
    
    images = [Image.open(image_path) for image_path in img_path_list]
    inputs = processor(images=images, return_tensors="pt", size=resize)

    return inputs

def run_dino_inference(input_list, device='cuda'):
    dino_model = AutoModel.from_pretrained(
        pretrain, 
        device_map=None, #"auto",
    )
    dino_model = dino_model.to(device)
    for p in dino_model.parameters():
        p.ds_status = False 

    dino_inputs = []
    for i in range(len(input_list)):
        inp = input_list[i]["pixel_values"]
        dino_inputs.append(inp)

    input_tensor = torch.stack(dino_inputs, dim=0)
    input_reshaped = input_tensor.view(-1, *input_tensor.shape[2:])
    input_reshaped = torch.tensor(input_reshaped, dtype=torch.bfloat16, device=device) #
    
    inputs = input_list[0]
    inputs["pixel_values"] = input_reshaped

    # import pdb; pdb.set_trace()  

    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16): # bfloat16
            outputs = dino_model(**inputs)

    return outputs.last_hidden_state[:, 5:, :]

class UpsampleWithPixelUnshuffle(nn.Module):
    def __init__(self, in_channels=3584, factor_1=2, factor_2=5, mid_dims=96):
        super().__init__()
        self.factor_1 = factor_1
        self.factor_2 = factor_2
        self.in_channels = in_channels
        self.mid_dims = mid_dims

        self.linear_1 = nn.Linear(self.in_channels // (self.factor_1 ** 2), self.mid_dims * (self.factor_2 ** 2))

    def forward(self, x):  # x: [B, 3584, 2, 4]
        B = x.shape[0]
        x_1 = F.pixel_shuffle(x, upscale_factor=self.factor_1)  # -> [B, 896, 4, 8]
        x_trans_1 = x_1.flatten(2, 3).permute(0, 2, 1) 

        x_2 = self.linear_1(x_trans_1).permute(0, 2, 1) 
        x_2_trans = x_2.view(B, -1, x_1.shape[2], x_1.shape[3]) # [B, 96*25, 4, 8]

        x_3 = F.pixel_shuffle(x_2_trans, upscale_factor=self.factor_2)  # -> [B, 96, 20, 40]
        x_out = x_3.flatten(2, 3).permute(0, 2, 1) # [B, 800, 96]

        return x_out

class Dinodecoer(nn.Module):
    def __init__(
        self,
        n_imgs=3,
        patch_w=4,
        patch_h=2,
        embed_dims=3584,
        patch_size=16,
        mid_dims=96,
        out_dims=1280,
        factor_1=4,
        factor_2=4,
        w_dino=0.5,
        mse_loss=True
    ):
        super().__init__()
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.n_imgs = n_imgs
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.mid_dims = mid_dims
        self.out_dims = out_dims
        self.H = self.patch_h * self.patch_size
        self.W = self.patch_w * self.patch_size
        self.factor_1 = factor_1
        self.factor_2 = factor_2
        self.w_dino = w_dino
        self.mse_loss = mse_loss

        self.feat_unsample_dino = UpsampleWithPixelUnshuffle(in_channels=self.embed_dims, factor_1=self.factor_1, factor_2=self.factor_2, mid_dims=self.mid_dims)
        self.linear_out = nn.Linear(self.mid_dims, self.out_dims)

    def forward(self, hidden_states, dino_feat): # [B, 24, 3584]  [B*3, 2450, 4096]
        B = hidden_states.shape[0]
        dim_dino = dino_feat.shape[-1]

        hidden_states = hidden_states.view(B, self.n_imgs, self.patch_h, self.patch_w, -1) # [B, 3, 2, 4, 3584]
        hidden_states = hidden_states.permute(0, 1, 4, 2, 3)
        hs = hidden_states.clone().view(B * self.n_imgs, -1, self.patch_h, self.patch_w) # [B*3, 3584, 2, 4]

        hs_up = self.feat_unsample_dino(hs)  # [B*3, 800, 96]
        hs_out = self.linear_out(hs_up) ## [B*3, 2048, 4096]

        if hs_out.shape[1] != dino_feat.shape[1]:
            dino_h = int((dino_feat.shape[1] // 2) ** 0.5)
            hs_out = hs_out.view(hs_out.shape[0], self.H, -1, hs_out.shape[-1]).permute(0, 3, 1, 2)
            hs_out = F.interpolate(hs_out, size=(dino_h, 2 * dino_h), mode='bilinear', align_corners=False)
            hs_out = hs_out.permute(0, 2, 3, 1).flatten(1,2)
            # dino_feat = dino_feat.view(dino_feat.shape[0], dino_h, -1, dino_feat.shape[-1]).permute(0, 3, 1, 2)
            # dino_feat = F.interpolate(dino_feat, size=(self.H, self.W), mode='bilinear', align_corners=False)
            # dino_feat = dino_feat.permute(0, 2, 3, 1).flatten(1,2)

        if self.mse_loss:
            dino_mse_loss = F.mse_loss(hs_out, dino_feat, reduction='mean')
            dist_loss = self.w_dino * dino_mse_loss  
        else:
            cos_sim = F.cosine_similarity(hs_out, dino_feat, dim=-1)  # [B*S, 512]
            cos_loss = 1 - cos_sim.mean()
            dist_loss = self.w_dino * cos_loss

        return dist_loss



class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        x = x.reshape(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, H, L, D)

    def _merge_heads(self, x):
        # x: (B, H, L, D)
        B, H, L, D = x.shape
        x = x.transpose(1, 2).reshape(B, L, H * D)
        return x

    def forward(self, query, key, value, attn_mask=None):
        """
        query: (B, Lq, C)
        key:   (B, Lk, C)
        value: (B, Lk, C)
        """
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        # Q,K,V: (B, H, L, D)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # (B,H,Lq,Lk)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # broadcast OK

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, H, Lq, D)

        attn_output = self._merge_heads(attn_output)  # (B, Lq, C)

        return self.out_proj(attn_output)

class Dinodecoer_MHCA_projector(nn.Module):
    def __init__(
        self,
        n_imgs=3,
        img_w=1120,
        img_h=560,
        embed_dims=3584,
        patch_size=16,
        out_dims=1280,
        scale=2,
        w_dino=1,
        mse_loss=True
    ):
        super().__init__()
        self.n_imgs = n_imgs
        self.embed_dims = embed_dims
        self.out_dims = out_dims
        self.H = img_h // patch_size
        self.W = img_w // patch_size
        self.scale = scale
        self.mid_dims = self.out_dims // self.scale
        self.w_dino = w_dino
        self.mse_loss = mse_loss

        self.dino_projection = nn.Linear(self.embed_dims, self.mid_dims)
        self.dino_query_vectors = nn.Parameter(torch.randn(self.H * self.W, self.mid_dims, requires_grad=True))
        self.dino_cross_attention = CustomMultiheadAttention(
            embed_dim=self.mid_dims, 
            num_heads=8
        )

        self.dino_projection_out = nn.Linear(self.mid_dims, self.out_dims)

    def forward(self, hidden_states, dino_feat): # [B, 24, 3584]  [B*3, 32*64, 1280]
        B = hidden_states.shape[0]

        hidden_states = hidden_states.view(B * self.n_imgs, -1, self.embed_dims) # [B*3, 8, 3584]
        dino_proj = F.normalize(self.dino_projection(hidden_states), dim=-1)  # [B*3, 8, 1280 // s]

        dino_query = self.dino_query_vectors.unsqueeze(0).repeat(
            hidden_states.size(0), 1, 1
        )

        dino_attn_output = self.dino_cross_attention(
            query=dino_query,       
            key=dino_proj,
            value=dino_proj
        )

        hs_embed = self.dino_projection_out(dino_attn_output)

        if self.mse_loss:
            dino_mse_loss = F.mse_loss(hs_embed, dino_feat, reduction='mean')
            dist_loss = self.w_dino * dino_mse_loss  
        else:
            cos_sim = F.cosine_similarity(hs_embed, dino_feat, dim=-1)  # [B*S, 512]
            cos_loss = 1 - cos_sim.mean()
            dist_loss = self.w_dino * cos_loss

        return dist_loss