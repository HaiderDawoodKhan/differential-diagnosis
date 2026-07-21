#!/usr/bin/env bash
set -euo pipefail

# Uncomment the model scripts you want to run.
# python3 canary_qwen_2_5b.py --split both
python3 kyutai_stt_2_6b_en.py --split both
# python3 granite_speech_3_3_2b.py --split both
# python3 omniasr_llm_7b_v2.py --model-card omniASR_LLM_7B_v2 --lang eng_Latn --split both
# python3 omniasr_llm_7b_v2.py --model-card omniASR_CTC_7B_v2 --lang none --split both
