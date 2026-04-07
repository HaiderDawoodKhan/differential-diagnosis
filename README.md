# Healthcare Audio Pipeline

This repository contains a two-stage workflow:

1. Generate corrupted consultation audio from clean source files.
2. Run Canary ASR on both noisy and clean generated audio to produce transcripts.

## Repository Paths Used

- Clean/noisy generated audio root: `code/generated_audios`
- Noisy generated audio: `code/generated_audios/noisy`
- Clean generated audio: `code/generated_audios/clean`
- Audio corruption script: `code/audio_corruption/merge_corrupt.py`
- ASR script: `code/asr/models/canary_2.py`
- Segment annotations: `code/asr/pyannote_textgrid`
- Transcript output root: `code/asr/generated_transcripts`

## 1) Generate Corrupted Audio

Run:

```bash
python audio_corruption/merge_corrupt.py --clean-dir /absolute/path/to/clean_audio
```

With current defaults, you only need `--clean-dir`.

### merge_corrupt defaults

- `--noise-dir`: `code/audio_corruption/noise_library`
- `--out-dir`: `code/generated_audios`
- `--snrs`: `-2 3 8 13 18`
- `--sample-rate`: `16000`
- `--copy-clean`: enabled by default
- `--no-random-noise-start`: enabled by default
- `--blacklist-file`: `code/audio_corruption/blacklist.txt` if present

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
code/generated_audios/
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

- `code/generated_audios/noisy`
- `code/generated_audios/clean`

It matches consultation IDs to segment files under:

- `code/asr/pyannote_textgrid`

It writes transcripts to:

- `code/asr/generated_transcripts/<model_name>/...`

Current model list in script:

- `nvidia/canary-qwen-2.5b`

### Transcript output structure

```text
code/asr/generated_transcripts/
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

