"""Fine-tune SpeechT5 TTS on a subset of prepared LJSpeech data.

Example:
    # smoke test (~minutes)
    python src/train.py --name smoke --hours 0.1 --steps 20 --eval_every 10
    # real run
    python src/train.py --name ft_30m_lr1e-5 --hours 0.5 --steps 1000 --lr 1e-5
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (SpeechT5ForTextToSpeech, SpeechT5Processor,
                          get_linear_schedule_with_warmup)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def take_hours(items, hours):
    out, total = [], 0.0
    for it in items:
        if total >= hours * 3600:
            break
        out.append(it)
        total += it["dur"]
    print(f"Using {len(out)} clips = {total / 60:.1f} min")
    return out


def make_collate(pad_id, reduction):
    def collate(batch):
        B = len(batch)
        max_t = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((B, max_t), pad_id, dtype=torch.long)
        attn = torch.zeros((B, max_t), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = torch.tensor(b["input_ids"])
            attn[i, :n] = 1

        lens = [b["labels"].shape[0] for b in batch]
        max_l = max(lens)
        max_l -= max_l % reduction  # length must be divisible by reduction factor
        labels = torch.full((B, max_l, 80), -100.0)
        for i, b in enumerate(batch):
            l = min(lens[i], max_l)
            labels[i, :l] = b["labels"][:l]

        spk = torch.stack([b["spk"] for b in batch])
        return {"input_ids": input_ids, "attention_mask": attn,
                "labels": labels, "speaker_embeddings": spk}
    return collate


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name")
    ap.add_argument("--hours", type=float, default=0.5, help="training data size")
    ap.add_argument("--steps", type=int, default=1000, help="optimizer steps")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="data/processed")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print("Device:", device)

    train_all = torch.load(f"{args.data_dir}/train.pt")
    val = torch.load(f"{args.data_dir}/val.pt")
    train = take_hours(train_all, args.hours)

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    model.config.use_cache = False
    if args.freeze_encoder:
        for p in model.speecht5.encoder.parameters():
            p.requires_grad = False
    model.to(device).train()

    collate = make_collate(processor.tokenizer.pad_token_id, model.config.reduction_factor)
    train_loader = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=args.bs, shuffle=False, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, args.warmup, args.steps)

    run_dir = Path("checkpoints") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f"runs/{args.name}")
    history = {"args": vars(args), "train": [], "val": []}

    base_val = evaluate(model, val_loader, device)
    print(f"step 0 | val loss {base_val:.4f} (pretrained baseline)")
    history["val"].append((0, base_val))
    writer.add_scalar("val/loss", base_val, 0)

    batches = infinite(train_loader)
    best = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        running = 0.0
        for _ in range(args.accum):
            batch = {k: v.to(device) for k, v in next(batches).items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        history["train"].append((step, running))
        writer.add_scalar("train/loss", running, step)
        writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
        if step % 10 == 0:
            print(f"step {step} | train {running:.4f} | {(time.time() - t0) / 60:.1f} min")

        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate(model, val_loader, device)
            history["val"].append((step, v))
            writer.add_scalar("val/loss", v, step)
            print(f"step {step} | val loss {v:.4f}")
            if v < best:
                best = v
                model.save_pretrained(run_dir / "best")
                processor.save_pretrained(run_dir / "best")

    model.save_pretrained(run_dir / "last")
    history["best_val"] = best
    history["train_minutes"] = (time.time() - t0) / 60
    Path("results").mkdir(exist_ok=True)
    with open(f"results/{args.name}_history.json", "w") as f:
        json.dump(history, f)
    print(f"Done. best val {best:.4f}. Checkpoints in {run_dir}/")


if __name__ == "__main__":
    main()