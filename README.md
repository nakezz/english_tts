# English Text-to-Speech (TTS) Fine-Tuning with SpeechT5

This repository contains the coursework project for **Machine Learning (F.CSM308)** at the **School of Information and Communication Technology, MUST**.

The project implements an end-to-end Text-to-Speech (TTS) pipeline that fine-tunes Microsoft's [SpeechT5](https://arxiv.org/abs/2110.07205) model on the single-speaker [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) dataset, evaluates speech intelligibility with OpenAI Whisper, and measures speaker voice fidelity using SpeechBrain x-vectors.

---

## Architecture & Overview

```
                      +-------------------+
                      |   Text Input      |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      | SpeechT5 Encoder  |
                      +---------+---------+
                                |
+-------------------+           v
| Speaker X-Vector  | ---> +-------------------+
|  (SpeechBrain)    |      | SpeechT5 Decoder  |
+-------------------+      +---------+---------+
                                |
                                v (Mel-Spectrogram)
                      +-------------------+
                      |  HiFi-GAN Vocoder |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      | Audio Output Wave |
                      +-------------------+
```

- **Backbone Model:** [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) (Encoder-Decoder architecture)
- **Vocoder:** [microsoft/speecht5_hifigan](https://huggingface.co/microsoft/speecht5_hifigan)
- **Speaker Embedding:** 512-dimensional x-vector extracted via [speechbrain/spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)
- **Dataset:** [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) (~24 hours of English speech from a single female speaker)
- **Hardware Acceleration:** Native support for Apple Silicon GPU (`mps`) and standard `cpu`.

---

## Project Structure

```
.
├── checkpoints/              # Saved model checkpoints (best & last)
│   ├── ft_30m_lr1e-5/        # Fine-tuned on 30 min LJSpeech subset
│   └── smoke/                # Quick verification checkpoint
├── configs/                  # Configuration files
├── data/
│   ├── processed/            # Preprocessed PyTorch tensors (train.pt, val.pt, meta.pt)
│   ├── raw/                  # Downloaded raw LJSpeech-1.1 archive and audio
│   └── spkrec-xvect/         # Pretrained SpeechBrain x-vector weights
├── notebooks/                # Experimental and analysis notebooks
├── report/                   # LaTeX report assets
├── results/
│   ├── curves.png            # Training & validation loss curves
│   ├── eval.json             # Objective evaluation results (WER & speaker similarity)
│   └── *_history.json        # Training history logs
├── samples/
│   ├── baseline/             # Synthesized audio samples (Pretrained baseline)
│   └── ft_30m/               # Synthesized audio samples (Fine-tuned model)
├── src/
│   ├── prepare_data.py       # Data download, mel extraction & x-vector computation
│   ├── train.py              # Model fine-tuning script with PyTorch & Transformers
│   ├── generate.py           # Audio synthesis script using SpeechT5 + HiFi-GAN
│   ├── evaluate.py           # Evaluation script (Whisper WER + X-vector cosine similarity)
│   └── plot_curves.py        # Loss curve plotting utility
├── main.tex                  # Coursework report in LaTeX
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/username/english_tts.git
cd english_tts

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Preparation

Download LJSpeech-1.1, extract 80-channel log mel-spectrogram targets, and extract x-vector speaker embeddings:

```bash
python src/prepare_data.py --max_hours 3.0 --n_val 100 --max_sec 12.0
```

Outputs will be saved in `data/processed/`:
- `train.pt`: Processed training samples (tokens, mel targets, x-vector embeddings, duration).
- `val.pt`: 100 held-out validation samples.
- `meta.pt`: Speaker mean embedding and metadata.

---

## Training & Fine-Tuning

### Smoke Test (Quick sanity check)
Runs 20 optimizer steps on a 0.1-hour subset to verify pipeline integrity:

```bash
python src/train.py --name smoke --hours 0.1 --steps 20 --eval_every 10
```

### Main Fine-Tuning Run
Fine-tune SpeechT5 on a 30-minute subset (0.5 hours) for 1,000 steps with gradient accumulation:

```bash
python src/train.py --name ft_30m_lr1e-5 --hours 0.5 --steps 1000 --bs 4 --accum 8 --lr 1e-5 --warmup 100 --eval_every 100
```

Key Training Hyperparameters:
- **Optimizer:** AdamW (`lr = 1e-5`, `weight_decay = 0.0`)
- **Batch Size:** 4 micro-batch $\times$ 8 gradient accumulation steps = **32 effective batch size**
- **LR Schedule:** Linear warmup (100 steps) with linear decay
- **Gradient Clipping:** Max norm $1.0$

### Monitoring & Visualizations
Launch TensorBoard to monitor live metrics:

```bash
tensorboard --logdir runs/
```

Generate smoothed training and validation loss comparison plots:

```bash
python src/plot_curves.py
```
Output plot is saved to `results/curves.png`.

---

## Inference (Speech Synthesis)

Synthesize 10 benchmark sentences covering various phonetic nuances, numbers, and tongue-twisters:

**1. Generate with Pretrained Baseline:**
```bash
python src/generate.py --model microsoft/speecht5_tts --out samples/baseline
```

**2. Generate with Fine-Tuned Model:**
```bash
python src/generate.py --model checkpoints/ft_30m_lr1e-5/best --out samples/ft_30m
```

Generated `.wav` files are stored as `00.wav` through `09.wav` at 16 kHz sample rate.

---

## Evaluation

Evaluate audio intelligibility and speaker similarity using:
1. **Word Error Rate (WER):** Transcribes generated audio via OpenAI Whisper (`small.en`) and calculates WER using `jiwer`.
2. **Speaker Similarity:** Measures cosine similarity between the generated audio's x-vector and the ground truth LJSpeech mean speaker embedding.

```bash
python src/evaluate.py samples/baseline samples/ft_30m
```

Results are saved to `results/eval.json`.

---

## Experimental Results

| Metric | Pretrained Baseline | Fine-Tuned (`ft_30m_lr1e-5`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Validation Loss** | `0.5888` | **`0.4002`** | **-32.0%** |
| **Training Steps** | 0 | 1,000 steps (~41 min on Apple Silicon) | — |
| **Training Audio** | — | 30 minutes (0.5 hours) | — |

Validation loss dropped rapidly within the first 200 steps (from `0.5888` to `0.4104`), stabilizing to its optimum at step 800 (`0.4002`), demonstrating significant adaptation to the speaker's vocal characteristics and acoustic profile.

---

## Authors

- **A. Nurislam** 
- **G. Tsetsenbilguun**

Instructor: **J. Orgil**  
Course: **Machine Learning (F.CSM308)**  
School of Information and Communication Technology, Mongolian University of Science and Technology (MUST)

---

## References

1. Ao, J., et al. (2022). *SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing*. ACL 2022.
2. Ito, K. & Johnson, L. (2017). *The LJ Speech Dataset*. [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)
3. Kong, J., Kim, J., & Bae, J. (2020). *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis*. NeurIPS 2020.
4. Radford, A., et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. OpenAI.
5. Snyder, D., et al. (2018). *X-Vectors: Robust DNN Embeddings for Speaker Recognition*. ICASSP 2018.

