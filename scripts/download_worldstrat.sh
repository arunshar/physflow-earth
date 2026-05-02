#!/usr/bin/env bash
# Pull WorldStrat from Hugging Face Datasets and pre-tokenize to .pt tensors.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/worldstrat"
mkdir -p "$DATA"

python -c "
from datasets import load_dataset
ds = load_dataset('worldstrat/worldstrat')
ds.save_to_disk('$DATA/raw')
print('saved:', ds)
"

python -m physflow.scripts.tokenize_worldstrat --raw "$DATA/raw" --out "$DATA"
echo "Done."
