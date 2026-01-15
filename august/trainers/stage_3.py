import os, argparse, torch
from accelerate import Accelerator, DeepSpeedPlugin
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm
import wandb

# Disable DTensor before imports
from transformers import get_scheduler

from models.model import AUGUST

os.environ["TORCH_DISTRIBUTED_DISABLE_DTENSOR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from dataloaders.loader import MultiVQADataset


def custom_collate_fn(batch):
    return {
        "feature_path": [item["feature_path"] for item in batch],
        "features": [item["features"] for item in batch],
        "caption": [item.get("caption", None) for item in batch],
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "label": [item.get("label", None) for item in batch],
        "task": [item.get("task", None) for item in batch],
    }


def _base_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def training_step(model, batch, accelerator):
    base = _base_model(model)
    batch_size = len(batch["features"])
    g_loss = 0.0
    c_loss = 0.0
    t_loss = 0.0

    for i in range(batch_size):
        # Extract slide features per sample
        slide_feats, question_feature = base.get_slide_features(batch["features"][i], batch["question"][i])

        # Supervised loss
        agl_loss, acl_loss, stl_loss = base(
            slide_feats,
            batch["question"][i],
            question_feature,
            target_answer=batch["answer"][i],
            caption=batch["caption"][i],
            true_label=batch["label"][i],
            task_labels=batch["task"][i]
        )
        g_loss += agl_loss
        c_loss += acl_loss
        t_loss += stl_loss
    m_loss = base.acm_loss()
    g_loss = g_loss / batch_size
    c_loss = c_loss / batch_size
    t_loss = t_loss / batch_size
    loss = g_loss + c_loss + t_loss + m_loss
    return loss, g_loss, c_loss, t_loss, m_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Gastric VQA Stage-1 Training with Accelerate + DeepSpeed")

    parser.add_argument("--jsonl", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="/ssd4/khang/checkpoints/gastric_llama3_accelerate")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--run_name", type=str, default="gastric-vqa-stage1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default="",
                        help="Path to resume checkpoint if exists")
    return parser.parse_args()


def main():
    args = parse_args()

    # DeepSpeed plugin
    ds_plugin = DeepSpeedPlugin(zero_stage=2)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        deepspeed_plugin=ds_plugin
    )

    # seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = AUGUST(stage='stage_2')  # created on every process
    model.lora()

    if args.resume and os.path.exists(args.resume):
        if accelerator.is_main_process:
            state_dict = torch.load(args.resume, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        accelerator.wait_for_everyone()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = MultiVQADataset(args.jsonl)
    sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset.sample_weights))
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    total_training_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_training_steps * 0.03)  # warmup ratio = 0.1

    # cosine scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # wandb
    if accelerator.is_main_process:
        wandb.init(project="gastricVQA", name=args.run_name, config=vars(args))

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0

    model.train()
    accelerator.print(f"Starting training for {args.epochs} epochs")
    accelerator.print(f"Total batches per epoch: {len(dataloader)}")

    for epoch in range(args.epochs):
        epoch_loss, num_batches = 0.0, 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}",
                            disable=not accelerator.is_local_main_process)

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                loss, gen, coarse, token, margin = training_step(model, batch, accelerator)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()  # update learning rate
                optimizer.zero_grad()

                epoch_loss += gen.item()
                num_batches += 1
                global_step += 1

                if accelerator.is_main_process:
                    wandb.log({
                        "train/agl_loss": gen.item(),
                        "train/stl_loss": token.item(),
                        "train/acl_loss": coarse.item(),
                        "train/acm_loss": margin.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step
                    })

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        if accelerator.is_main_process:
            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch+1}, step=global_step)
            accelerator.print(f"Epoch {epoch+1} done. Avg loss: {avg_epoch_loss:.4f}")
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), f"{args.output_dir}/august_s3.pth")
            accelerator.print(f"Checkpoint saved. at {args.output_dir}/august_s3.pth")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()

    accelerator.print("Training finished!")


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
