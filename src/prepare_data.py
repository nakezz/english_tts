import argparse
import random
import tarfile
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import SpeechT5Processor

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
SR = 16000


def download(raw_dir: Path) -> Path:
    root = raw_dir / "LJSpeech-1.1"
    if root.exists():
        return root
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "LJSpeech-1.1.tar.bz2"
    if not archive.exists():
        print("Downloading LJSpeech (~2.6 GB)...")
        urllib.request.urlretrieve(URL, archive)
    print("Extracting...")
    with tarfile.open(archive) as tar:
        tar.extractall(raw_dir)
    return root


def load_metadata(root: Path):
    rows = []
    with open(root / "metadata.csv", encoding="utf-8") as f:
        for line in f:
            uid, _raw, norm = line.rstrip("\n").split("|")
            rows.append((uid, norm))  # normalized text: numbers already spelled out
    return rows


def load_speaker_model():
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:  # older speechbrain
        from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="data/spkrec-xvect",
        run_opts={"device": "cpu"},
    )


def featurize(items, root, processor, spk_model, max_sec, max_hours=None, desc=""):
    out, total = [], 0.0
    for n, (uid, text) in enumerate(items):
        if max_hours is not None and total >= max_hours * 3600:
            break
        wav, _ = librosa.load(root / "wavs" / f"{uid}.wav", sr=SR)
        dur = len(wav) / SR
        if dur > max_sec:
            continue
        feats = processor.feature_extractor(audio_target=wav, sampling_rate=SR, return_attention_mask=False)
        mel = feats["input_values"][0]
        with torch.no_grad():
            emb = spk_model.encode_batch(torch.from_numpy(wav).unsqueeze(0))
            emb = torch.nn.functional.normalize(emb, dim=2).reshape(-1)
        out.append({
            "id": uid,
            "text": text,
            "input_ids": processor.tokenizer(text)["input_ids"],
            "labels": torch.as_tensor(np.asarray(mel, dtype=np.float32)),  # (T, 80)
            "spk": emb.float(),  # (512,)
            "dur": dur,
        })
        total += dur
        if len(out) % 100 == 0:
            print(f"[{desc}] {len(out)} clips, {total / 3600:.2f} h")
    print(f"[{desc}] done: {len(out)} clips, {total / 3600:.2f} h")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_hours", type=float, default=3.0, help="max training audio to prepare")
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()

    root = download(Path(args.raw_dir))
    items = load_metadata(root)
    random.Random(args.seed).shuffle(items)
    val_items, train_items = items[: args.n_val], items[args.n_val:]

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    spk_model = load_speaker_model()

    val = featurize(val_items, root, processor, spk_model, args.max_sec, desc="val")
    train = featurize(train_items, root, processor, spk_model, args.max_sec,
                      max_hours=args.max_hours, desc="train")

    meta = {
        "mean_spk": torch.stack([t["spk"] for t in train]).mean(0, keepdim=True),
        "n_train": len(train),
        "n_val": len(val),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(train, out / "train.pt")
    torch.save(val, out / "val.pt")
    torch.save(meta, out / "meta.pt")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()