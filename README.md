# Healthcare Audio Pipeline

This repository contains a two-stage workflow:

1. Generate corrupted consultation audio from clean source files.
2. Run Canary ASR on both noisy and clean generated audio to produce transcripts.

## Repository Paths Used

- Clean/noisy generated audio root: `generated_audios`
- Noisy generated audio: `generated_audios/noisy`
- Clean generated audio: `generated_audios/clean`
- Audio corruption script: `audio_corruption/merge_corrupt.py`
- ASR script: `asr/models/canary_2.py`
- Segment annotations: `asr/pyannote_textgrid`
- Transcript output root: `asr/generated_transcripts`

## Noise Library Setup 

The repository does not include the noise library folder.

Download it from Google Drive:

- https://drive.google.com/file/d/1RmLVmpwi-L2uTU7KVpLUnOWpLk2ojaGs/view?usp=sharing

Then unzip and place the folder here:

- `audio_corruption/noise_library`

Quick steps:

1. Open the link and download the archive.
2. Extract/unzip it.
3. Rename the extracted folder to `noise_library` if needed.
4. Move it into `audio_corruption/` so the final path is `audio_corruption/noise_library/`.

Without this folder, `merge_corrupt.py` cannot generate noisy outputs.

## 1) Generate Corrupted Audio

Run:

```bash
python audio_corruption/merge_corrupt.py --clean-dir /absolute/path/to/clean_audio
```

With current defaults, you only need `--clean-dir`.
The experiments were conducted on the Primock57 dataset.

### merge_corrupt defaults

- `--noise-dir`: `audio_corruption/noise_library`
- `--out-dir`: `generated_audios`
- `--snrs`: `-2 3 8 13 18`
- `--sample-rate`: `16000`
- `--copy-clean`: enabled by default
- `--no-random-noise-start`: enabled by default
- `--blacklist-file`: `audio_corruption/blacklist.txt` if present

### Blacklist behavior

If blacklist file is present (or provided), consultation IDs in it are skipped entirely.

Example lines:

```text
day1_consultation11
day1_consultation13
day2_consultation02, day2_consultation08
```

Skipped consultations are not merged, not corrupted, and do not appear in outputs.

### Generated structure

```text
    /generated_audios/
	clean/
		dayX_consultationYY.wav
	noisy/
		<category>/
			snr_-2/
			snr_3/
			snr_8/
			snr_13/
			snr_18/
	metadata.csv
	README.txt
```

## 2) Generate ASR Transcripts (Canary)

Run:

```bash
python asr/models/canary_2.py
```

The script reads audio from:

- `generated_audios/noisy`
- `generated_audios/clean`

It matches consultation IDs to segment files under:

- `asr/pyannote_textgrid`

It writes transcripts to:

- `asr/generated_transcripts/<model_name>/...`

Current model list in script:

- `nvidia/canary-qwen-2.5b`

### Transcript output structure

```text
asr/generated_transcripts/
	canary-qwen-2.5b/
		noisy/
			<category>/snr_<value>/dayX_consultationYY.txt
		clean/
			dayX_consultationYY.txt
```

## Environment Notes

`asr/models/canary_2.py` requires Python packages including at least:

- torch
- soundfile
- numpy
- librosa
- nemo (SpeechLM2/SALM support)

If your editor reports unresolved imports, install packages in the active Python environment before running.

