"""Synthesize test sentences with a SpeechT5 model (baseline or fine-tuned).

Examples:
    python src/generate.py --model microsoft/speecht5_tts --out samples/baseline
    python src/generate.py --model checkpoints/ft_30m_lr1e-5/best --out samples/ft_30m
"""
import argparse
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Printing, in the only sense with which we are at present concerned, differs from most arts.",
    "Did you hear what happened at the meeting yesterday?",
    "I can't believe we actually won the championship!",
    "The train leaves at seven thirty in the morning.",
    "In nineteen ninety nine, the company earned two million dollars.",
    "Please call doctor Smith at the clinic on Main Street.",
    "Peter Piper picked a peck of pickled peppers.",
    "Machine learning models learn patterns from large amounts of data.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/speecht5_tts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="data/processed/meta.pt")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained(args.model).to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    spk = torch.load(args.meta)["mean_spk"].to(device)  # (1, 512)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(SENTENCES):
        ids = processor(text=text, return_tensors="pt")["input_ids"].to(device)
        t0 = time.time()
        with torch.no_grad():
            wav = model.generate_speech(ids, spk, vocoder=vocoder)
        elapsed = time.time() - t0
        wav = wav.cpu().numpy()
        sf.write(out / f"{i:02d}.wav", wav, 16000)
        print(f"{i:02d} | RTF {elapsed / (len(wav) / 16000):.2f} | {text[:50]}")


if __name__ == "__main__":
    main()