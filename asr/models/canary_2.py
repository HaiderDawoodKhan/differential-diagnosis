import os
import re
import torch
import soundfile as sf
import numpy as np
import librosa

from pathlib import Path
from dataclasses import dataclass
from typing import List

# 🔥 NEW IMPORT (replace Parakeet)
from nemo.collections.speechlm2.models import SALM

# ================= PATH SETUP ================= #

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASR_ROOT = PROJECT_ROOT / "asr"

GENERATED_AUDIOS_ROOT = PROJECT_ROOT / "generated_audios"
NOISY_ROOT = GENERATED_AUDIOS_ROOT / "noisy"
CLEAN_ROOT = GENERATED_AUDIOS_ROOT / "clean"
TEXTGRID_FOLDER = ASR_ROOT / "pyannote_textgrid"
OUTPUT_ROOT = ASR_ROOT / "generated_transcripts"

# ================= CONFIG ================= #

MODELS = [
    "nvidia/canary-qwen-2.5b",
]

TARGET_SR = 16000
TMP_FILE = str(BASE_DIR / "tmp_segment.wav")

# ================= DEVICE ================= #

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ================= DATA STRUCT ================= #

@dataclass
class SpeechSegment:
    start: float
    end: float
    speaker: str

# ================= ID EXTRACTION ================= #

def extract_conv_id(name: str):
    name = name.lower()
    match = re.search(r"(day\d+_consultation\d+)", name)
    return match.group(1) if match else None

# ================= PARSER ================= #

def parse_segments(file_path: Path) -> List[SpeechSegment]:
    segments = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(",")

        if len(parts) != 3:
            continue

        try:
            start, end, speaker = parts
            segments.append(
                SpeechSegment(
                    start=float(start),
                    end=float(end),
                    speaker=speaker,
                )
            )
        except:
            continue

    return segments

# ================= INDEX ================= #

def build_index():
    index = {}

    for root, _, files in os.walk(TEXTGRID_FOLDER):
        for file in files:
            if file.endswith(".csv") or file.endswith(".TextGrid"):
                conv_id = extract_conv_id(file)

                if not conv_id:
                    continue

                full_path = Path(root) / file

                if conv_id not in index:
                    index[conv_id] = []

                index[conv_id].append(full_path)

    print(f"\nIndexed {len(index)} conversations")

    print("\nSample index:")
    for k in list(index.keys())[:5]:
        print(k, "->", len(index[k]), "files")

    return index

# ================= AUDIO ================= #

def extract_segment(audio, sr, start, end):
    s = int(start * sr)
    e = min(int(end * sr), len(audio))

    if s >= len(audio):
        return None

    segment = audio[s:e]

    if len(segment) < 0.1 * sr:
        return None

    sf.write(TMP_FILE, segment.astype(np.float32), sr)
    return TMP_FILE

# ================= ASR ================= #

def load_model(name):
    print(f"\nLoading model: {name}")

    model = SALM.from_pretrained(name)

    model = model.to(device)

    # 🔥 IMPORTANT for RTX 4060 (8GB)
    model = model.half()

    model.eval()
    return model


@torch.inference_mode()
def transcribe(model, audio_file):
    try:
        outputs = model.generate(
            prompts=[
                [{
                    "role": "user",
                    "content": f"Transcribe: {model.audio_locator_tag}",
                    "audio": [audio_file]
                }]
            ],
            max_new_tokens=128,
        )

        out = outputs[0]

        if isinstance(out, torch.Tensor):
            text = model.tokenizer.ids_to_text(out.cpu())
        else:
            text = str(out)

        return text.strip()

    except Exception as e:
        print(f"[TRANSCRIBE ERROR] {e}")
        return ""

# ================= CORE ================= #

def process_file(model, audio_path, segments):
    results = []

    # 🔥 LOAD AUDIO ONCE (BIG SPEEDUP)
    data, sr = sf.read(audio_path)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    for seg in segments:
        tmp = extract_segment(data, sr, seg.start, seg.end)
        if not tmp:
            continue

        text = transcribe(model, tmp)

        results.append({
            "speaker": seg.speaker,
            "start": seg.start,
            "text": text
        })

    results.sort(key=lambda x: x["start"])

    return "\n".join([f"{r['speaker']}: {r['text']}" for r in results])

# ================= DRIVER ================= #

def run():
    index = build_index()

    all_wavs = []
    source_roots = [
        ("noisy", NOISY_ROOT),
        ("clean", CLEAN_ROOT),
    ]

    for source_name, source_root in source_roots:
        if not source_root.exists():
            print(f"[WARN] Missing audio folder: {source_root}")
            continue

        for root, _, files in os.walk(source_root):
            for f in files:
                if f.lower().endswith(".wav"):
                    all_wavs.append((source_name, source_root, Path(root), f))

    print(f"\nFound {len(all_wavs)} audio files")

    for model_name in MODELS:
        model = load_model(model_name)
        model_tag = model_name.split("/")[-1]

        processed = 0
        skipped = 0
        errors = 0

        for source_name, source_root, root, wav in all_wavs:
            conv_id = extract_conv_id(wav)

            if processed < 3:
                print("\n[DEBUG]")
                print("source:", source_name)
                print("wav:", wav)
                print("conv_id:", conv_id)

            if not conv_id or conv_id not in index:
                print(f"[NO MATCH] {wav}")
                skipped += 1
                continue

            seg_file = index[conv_id][0]
            segments = parse_segments(seg_file)

            if not segments:
                print(f"[EMPTY SEGMENTS] {seg_file}")
                skipped += 1
                continue

            rel_path = root.relative_to(source_root)
            output_dir = OUTPUT_ROOT / model_tag / source_name / rel_path
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{conv_id}.txt"

            if output_file.exists():
                skipped += 1
                continue

            audio_path = root / wav

            try:
                print(f"Processing: {audio_path}")

                convo = process_file(model, audio_path, segments)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(convo)

                processed += 1

            except Exception as e:
                errors += 1
                print(f"[ERROR] {wav}: {e}")

        print(f"\n=== Model {model_tag} DONE ===")
        print(f"Processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Errors: {errors}")

        del model
        torch.cuda.empty_cache()

    if os.path.exists(TMP_FILE):
        os.remove(TMP_FILE)

    print("\nALL DONE")

# ================= RUN ================= #

if __name__ == "__main__":
    run()