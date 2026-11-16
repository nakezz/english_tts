import argparse
import json
from pathlib import Path

import jiwer
import librosa
import torch
import whisper
from whisper.normalizers import EnglishTextNormalizer

from generate import SENTENCES
from prepare_data import load_speaker_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--whisper", default="small.en")
    ap.add_argument("--meta", default="data/processed/meta.pt")
    args = ap.parse_args()

    asr = whisper.load_model(args.whisper, device="cpu")
    norm = EnglishTextNormalizer() 
    spk_model = load_speaker_model()
    ref_spk = torch.nn.functional.normalize(
        torch.load(args.meta)["mean_spk"].reshape(-1), dim=0)

    results = {}
    for d in args.dirs:
        refs, hyps, sims = [], [], []
        for i, text in enumerate(SENTENCES):
            path = Path(d) / f"{i:02d}.wav"
            hyp = asr.transcribe(str(path), language="en", fp16=False)["text"]
            refs.append(norm(text))
            hyps.append(norm(hyp))
            wav, _ = librosa.load(path, sr=16000)
            with torch.no_grad():
                emb = spk_model.encode_batch(torch.from_numpy(wav).unsqueeze(0)).reshape(-1)
            sims.append(torch.nn.functional.cosine_similarity(emb, ref_spk, dim=0).item())

        wer = jiwer.wer(refs, hyps)
        sim = sum(sims) / len(sims)
        results[d] = {"wer": wer, "speaker_sim": sim}
        print(f"\n=== {d} ===\nWER: {wer:.3f} | speaker sim: {sim:.3f}")
        for r, h in zip(refs, hyps):
            if r != h:
                print(f"  ref: {r}\n  hyp: {h}")

    Path("results").mkdir(exist_ok=True)
    with open("results/eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/eval.json")


if __name__ == "__main__":
    main()