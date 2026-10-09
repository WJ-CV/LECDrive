#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$PYTHONPATH
TIME_STR=$(date +"%m_%d_%H_%M_%S")

# env for multiple nodes running
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_PORT=${MLP_WORKER_0_PORT:-$(shuf -i 10000-50000 -n1)}
MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}
NUM_GPUS=$(nvidia-smi -L | wc -l)

DATAROOT="/e2e-data/evad-osc-datasets"

BS_TRAIN=8
BS_TEST=1
DATASET="NAVSIM"
PROMPT_TYPE="angle_speed" #""

INFER_JSON="cache_files/navsim_cache/navsim_test_4s_3v_1f_yaw_system_user_prompt.json"
MODEL_NAME_OR_PATH="outputs_navsim/Qwen2.5-VL-7B-Instruct/dino3_dim224_mse0.5_sam3_w0.5_da3_60m_w1.5_8token-lr5e-5"
OUTDIR="outputs_navsim/navsim_test/dino3_dim224_mse0.5_sam3_w0.5_da3_60m_w1.5_8token-lr5e-5"

function evaluate(){
  torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    open_r1/recipes/inference_rank_save_CoVT_dinov3_sam3_da3_lidardepth_motiontoken.py \
    --use_dinov3 True \
    --use_sam3 True \
    --use_depth True \
    --output_dir $OUTDIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_classname NAVSIM \
    --dataset_name ${DATAROOT}/datasets/NAVSIM \
    --img_weight 1920 \
    --img_height 1080 \
    --re_weight 1120 \
    --re_height 560 \
    --num_views 3 \
    --meta_dir /e2e-data/users/lg/workspace/users/zhangyikai/meta_data/navsim \
    --attn_implementation flash_attention_2 \
    --cache_file $INFER_JSON \
    --rebuild False \
    --batch_size $BS_TEST \
    --bf16 \
    --split val \
    --version v1.0-trainval \
    --max_new_tokens 4096 \
    --query_prompt_type $PROMPT_TYPE \
    2>&1 | tee -a $OUTDIR/evalation_logs.txt
}
evaluate
