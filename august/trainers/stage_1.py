import os, argparse, torch
from accelerate import Accelerator
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, error_handler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import wandb

os.environ["TORCH_DISTRIBUTED_DISABLE_DTENSOR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # or "true"

from transformers import get_scheduler
from models.model import AUGUST
from dataloaders.loader import SingleQADataset 
from accelerate import Accelerator, DeepSpeedPlugin


def custom_collate_fn(batch):
    return {
        "features": [item["features"] for item in batch],
        # "caption": [item.get("caption", None) for item in batch],
        "coords_path": [item.get("coords_path", None) for item in batch],
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "label": [item.get("label", None) for item in batch],
        "task": [item.get("task", None) for item in batch],
    }


def _base_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def training_step(model, batch):
    base = _base_model(model)
    batch_size = len(batch["features"])

    # fallback to per-sample
    loss_sum = 0.0
    for i in range(batch_size):
        slide_feat, question_feature = base.get_slide_features(batch["features"][i], batch["question"][i])
        loss = base(slide_feat, batch["question"][i], question_feature, target_answer=batch["answer"][i])
        loss_sum += loss
    return loss_sum / batch_size


def parse_args():
    parser = argparse.ArgumentParser(description="Gastric VQA Stage-0 Training with Accelerate")

    # Data arguments
    parser.add_argument("--jsonl", type=str,
                        default="./dataset_csv/CMCUJB_CAP_train_v7.jsonl",
                        help="Path to training JSONL file")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")

    # Output arguments
    parser.add_argument("--output_dir", type=str,
                        default="/ssd4/khang/checkpoints/gastric_llama3_accelerate",
                        help="Output directory for checkpoints")

    parser.add_argument("--run_name", type=str, default="gastric-vqa-stage0",
                        help="W&B run name")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num workers")

    return parser.parse_args()


def main():
    args = parse_args()
    ds_plugin = DeepSpeedPlugin(zero_stage=2)
    # Accelerator (do NOT call dist.init_process_group yourself when using accelerate)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        deepspeed_plugin=ds_plugin
    )

    # seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = AUGUST(stage='stage_1')  # created on every process
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = SingleQADataset(args.jsonl, task='caption')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # total training steps = epochs × steps_per_epoch
    total_training_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_training_steps * 0.03)  # warmup ratio = 0.03

    # cosine scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Now init wandb only on main process
    if accelerator.is_main_process:
        wandb.init(
            project="gastricVQA",
            name=args.run_name,
            config=vars(args)
        )

    # Prepare with accelerate (same objects on all ranks)
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # create output_dir on main process
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    model.train()
    accelerator.print(f"Starting training for {args.epochs} epochs...")
    accelerator.print(f"Total batches per epoch: {len(dataloader)}")
    best_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}",
                            disable=not accelerator.is_local_main_process)

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                loss = training_step(model, batch)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()  # update learning rate
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if accelerator.is_main_process:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step
                    })

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        if accelerator.is_main_process:
            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch + 1}, step=global_step)
            accelerator.print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), f"{args.output_dir}/august_s1.pth")
                accelerator.print(f"Best model saved with loss {best_loss:.4f} at epoch {epoch + 1}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()

    accelerator.print("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, sys, os

        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
        print(f"\n\n*** Exception caught on rank {rank} ***\n", file=sys.stderr)
        traceback.print_exc()
        # ensure other procs don't hang waiting for collectives
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        raise
