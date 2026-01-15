export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=6,7

accelerate launch --config_file ./august/configs/configs.yaml \
    ./august/trainers/stage_3.py \
    --jsonl ./august/data/CMCUJB_VQA_train.jsonl \
    --epochs 20 \
    --lr 2e-4 \
    --gradient_accumulation_steps 1 \
    --output_dir "checkpoint_path"  \
    --run_name gastric-vqa-stage3 \
    --resume "stage_2_path" \
    --batch_size 1
