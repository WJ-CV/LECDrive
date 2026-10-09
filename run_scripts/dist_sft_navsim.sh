#!/bin/bash      
# source /openr1/bin/activate

export PYTHONPATH=$(pwd):$PYTHONPATH
TIME_STR=$(date +"%m_%d_%H_%M_%S")
USE_DINOV3=${USE_DINOV3:-False}
USE_SAM3=${USE_SAM3:-False}
USE_DEPTH=${USE_DEPTH:-False}
CONFIG=${CONFIG:-"baseline"}

DATAROOT="/e2e-data/evad-osc-datasets"
OUTDIR="$(pwd)/outputs_navsim/Qwen2.5-VL-7B-Instruct/${TIME_STR}_${CONFIG}" # 8dino_msew1_8sam_w0.5_32da_l1w0.1
META_DATA_DIR="/e2e-data/users/lg/workspace/users/zhangyikai/meta_data"
mkdir -p $OUTDIR
START_INTERVAL=15

NUM_GPUS=$(nvidia-smi -L | wc -l)
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
PORT=${MLP_WORKER_0_PORT:-$(shuf -i 10000-50000 -n1)}
MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}

DINO_pretrain="/e2e-data/evad-tech-vla/wangjie68/experiments-VLA/dinov3/pretrain/dinov3-vith16plus-pretrain-lvd1689m" #  dinov3-vit7b16-pretrain-lvd1689m  dinov3-vith16plus-pretrain-lvd1689m
SAM_pretrain="/e2e-data/evad-tech-vla/wangjie68/experiments-VLA/samv3/pretrain/sam3/sam3.pt"
TRAIN_JSON="cache_files/navsim_cache/navsim_train_4s_3v_1f_yaw_system_user_prompt.json"

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    open_r1/recipes/sft_CoVT_dinov3_sam3_da3_lidardepth_motiontoken.py \
    --use_dinov3 $USE_DINOV3 \
    --w_dinov3 0.5 \
    --use_sam3 $USE_SAM3 \
    --w_sam3 0.5 \
    --use_depth $USE_DEPTH \
    --w_da3 1.5 \
    --deepspeed local_scripts/zero3.json \
    --output_dir $OUTDIR \
    --model_name_or_path ${DATAROOT}/pretrained/Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_classname NAVSIM \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --dataset_name ${DATAROOT}/datasets/NAVSIM \
    --meta_dir $META_DATA_DIR/navsim/ \
    --rebuild True \
    --split train \
    --logging_steps 1 \
    --bf16 \
    --version v1.0-trainval \
    --img_weight 1920 \
    --img_height 1080 \
    --re_weight 1120 \
    --re_height 560 \
    --num_views 3 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing True \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 5 \
    --run_name $TIME_STR \
    --save_steps 2000 \
    --save_only_model true \
    --save_total_limit 1 \
    --max_seq_length 4096 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --lr_scheduler_type cosine \
    --max_grad_norm 2.0 \
    --warmup_ratio 0.1 \
    --query_prompt_type html_state_nav \
    --ratio_shuffle False \
    --mixed_shuffle True \
    --dataloader_num_workers 4 \
    --frame_start_interval $START_INTERVAL \
    --force_load True \
    --cache_file $TRAIN_JSON \
    --dino_pretrain_path $DINO_pretrain \
    --sam_pretrain_path $SAM_pretrain \
    2>&1 | tee -a $OUTDIR/training_logs.txt