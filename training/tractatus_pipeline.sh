#!/usr/bin/env bash
# The Tractatus LoRA pipeline, end to end: rows -> both LoRAs (Tractatus and answer-only control) -> GGUF.
# Defaults run the mock-up (training/mock.py, CPU, seconds per step). The real run only changes variables:
#   BASE=alibiserikbay/JevK5 DATA=../data/tractatus/train.jsonl OUT=../data/lora DEVICE=cuda \
#   EPOCHS=1 LR=3e-5 RANK=16 WARMUP=20 OUTTYPE=q8_0 bash tractatus_pipeline.sh
# DATA may name several tractatus_data.py outputs (space-separated); REPLAY adds plain rows as they are. Evaluation runs afterwards against llama-server
# (bench/tractatus_eval.py, bench/tractatus_pilot.py).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
BASE=${BASE:-../data/mock/base}
DATA=${DATA:-../data/tractatus/smoke.jsonl}
OUT=${OUT:-../data/mock}
DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-60}
LR=${LR:-3e-3}
RANK=${RANK:-8}
WARMUP=${WARMUP:-5}
DEV_SHARE=${DEV_SHARE:-0.1}
OUTTYPE=${OUTTYPE:-f16}
REPLAY=${REPLAY:+--replay $REPLAY}
CONVERT=${CONVERT:-../data/llama.cpp/convert_hf_to_gguf.py}

[ -e "$BASE" ] || [[ "$BASE" == */* && ! "$BASE" == ../* ]] || $PY mock.py "$BASE"
for variant in tractatus answers; do
    $PY build_tractatus.py $DATA "$OUT/lora-$variant" --variant "$variant" --dev-share "$DEV_SHARE" $REPLAY
done
# Equal budget: the control trains exactly as many optimizer steps as the Tractatus LoRA.
steps=()
for variant in tractatus answers; do
    $PY lora.py "$OUT/lora-$variant" "$OUT/out-$variant" --base "$BASE" --device "$DEVICE" \
        --rank "$RANK" --lr "$LR" --epochs "$EPOCHS" --warmup "$WARMUP" --max-len 4096 "${steps[@]}"
    steps=(--steps "$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['total_steps'])" \
        "$OUT/out-tractatus/train_log.json")")
    $PY "$CONVERT" "$OUT/out-$variant" --outfile "$OUT/$variant-$OUTTYPE.gguf" --outtype "$OUTTYPE" --no-mtp
done
echo "done: $OUT/tractatus-$OUTTYPE.gguf, $OUT/answers-$OUTTYPE.gguf"
