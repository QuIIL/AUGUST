export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=

accelerate launch --config_file ./august/configs/configs.yaml \
    ./august/trainers/stage_1.py \
    --jsonl ./august/data/CMCUJB_VQA_train.jsonl \
    --epochs 40 \
    --lr 1e-3 \
    --gradient_accumulation_steps 1 \
    --output_dir "checkpoint_path" \
    --run_name gastric-vqa-stage2 \
    --resume "stage_1_path" \
    --batch_size 8
