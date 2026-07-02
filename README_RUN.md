# Product Type Signature 100k Debug Runner

## Locked settings

```python
PARALLEL_WORKERS = 10
SIGNATURES_PER_LLM_CALL = 5
EMBED_BATCH_SIZE = 16
```

## Flow

```text
product_description
  -> signature
  -> group by signature
  -> send signature only to Gemma 3 1B IT
  -> 5 signatures per LLM request
  -> BGE-small canonical mapping on CPU
  -> output product type + canonical id + canonical category
```

## Start Gemma vLLM

```bash
vllm serve google/gemma-3-1b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 2048 \
  --enable-prefix-caching \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384
```

## Install runner packages

```bash
pip install pandas numpy requests sentence-transformers
```

## Edit paths in script

Open `run_producttype_signature_100k_debug.py` and edit:

```python
INPUT_FILE = "/home/ubuntu/raw_100k.csv"
OUTPUT_DIR = "/home/ubuntu/producttype_run_001"
PRODUCT_DESC_COL = "product_description"
ONTOLOGY_SEED_CSV = "/home/ubuntu/ontology_seed.csv"
LLM_BASE_URL = "http://127.0.0.1:8000"
```

## Run

```bash
python run_producttype_signature_100k_debug.py
```

## Debug outputs

The script prints progress in terminal and saves debug files:

- `llm_batch_debug.csv`
  - which LLM batch ran
  - batch id out of total
  - exact signatures sent to Gemma
  - product types returned
  - raw response preview
  - parse status
  - elapsed seconds

- `canonical_batch_debug.csv`
  - which canonical embedding batch ran
  - batch id out of total
  - product type count
  - product type preview
  - elapsed seconds

- `run_summary.json`
  - total elapsed time
  - raw rows
  - junk rows
  - unique signatures
  - LLM batches
  - mapped/unmapped signatures

- `signature_producttype_map.csv`
  - one row per unique signature
  - LLM product type
  - canonical id/name/category
  - similarity/status
  - raw row count

- `producttype_output.csv`
  - final row-level output mapped back to every raw record
