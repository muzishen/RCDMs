export MODEL_DIR="./weights/prior_diffuser/kandinsky-2-2-prior"
export OUTPUT_DIR="./logs/stage1/FlintstonesSV"

accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 --use_deepspeed --mixed_precision="fp16"  train_stage1.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --seed 42 \
 --learning_rate=1e-5 \
 --train_batch_size=1 \
 --max_train_steps=1000000 \
 --mixed_precision="fp16" \
 --checkpointing_steps=5000  \
 --noise_offset=0.1 \
 --report_to=tensorboard \
 --num_warmup_steps 2000  \
 --config="./configs/training.yaml" \
 --dataset='flintstones' \
 --sr

