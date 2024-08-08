export MODEL_DIR="./weights/stable-diffusion-v1-5"
export OUTPUT_DIR="./logs/stage2/FlintstonesSV"

accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 --use_deepspeed \
  --deepspeed_config_file zero_stage2_config.json \
  train_stage2.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --learning_rate=1e-5 \
 --adam_weight_decay=0.01 \
 --train_batch_size=1 \
 --max_train_steps=1000000 \
 --checkpointing_steps=10000  \
 --noise_offset=0.1 \
 --report_to=tensorboard \
 --lr_warmup_steps=2000  \
 --config="./configs/training.yaml" \
 --dataset='flintstones' \
 --sr 


