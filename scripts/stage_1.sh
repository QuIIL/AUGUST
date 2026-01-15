export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=

accelerate launch --config_file ./august/configs/configs.yaml ./august/trainers/stage_0.py" \
    --jsonl ./august/data/CMCUJB_CAP_train.jsonl \
    --epochs 40 --batch_size 16 --lr 1e-3 \
    --gradient_accumulation_steps 1 \
    --output_dir "checkpoint_path" \
    --run_name gastric-vqa-stage1 \
