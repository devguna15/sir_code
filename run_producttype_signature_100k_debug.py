#!/usr/bin/env python3
"""
Run product type generation from product descriptions using signature grouping.

Flow:
1. Read raw CSV.
2. Build signature from product_description.
3. Group rows by signature.
4. Send signatures only to Gemma via vLLM, 5 signatures per request.
5. Map LLM product_type to canonical ontology using BAAI/bge-small-en-v1.5 on CPU.
6. Write output CSV plus debug CSVs.

Locked settings:
- PARALLEL_WORKERS = 10
- SIGNATURES_PER_LLM_CALL = 5
- EMBED_BATCH_SIZE = 16
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests

# =========================
# USER CONFIG - EDIT THESE
# =========================
INPUT_FILE = "/home/ubuntu/raw_100k.csv"
OUTPUT_DIR = "/home/ubuntu/producttype_run_001"
PRODUCT_DESC_COL = "product_description"

LLM_BASE_URL = "http://127.0.0.1:8000"
LLM_MODEL = "google/gemma-3-1b-it"

ONTOLOGY_SEED_CSV = "/home/ubuntu/ontology_seed.csv"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DEVICE = "cpu"

PARALLEL_WORKERS = 10
SIGNATURES_PER_LLM_CALL = 5
EMBED_BATCH_SIZE = 16
MAX_TOKENS = 256
TIMEOUT = 60
ONTOLOGY_THRESHOLD = 0.62

DEBUG_PRINT_SIGNATURES = True
DEBUG_SIGNATURE_PREVIEW_CHARS = 500

# =========================
# Output files
# =========================
OUTPUT_CSV = "producttype_output.csv"
SIGNATURE_MAP_CSV = "signature_producttype_map.csv"
UNMAPPED_CANDIDATES_CSV = "unmapped_candidates.csv"
RUN_SUMMARY_JSON = "run_summary.json"
LLM_DEBUG_CSV = "llm_batch_debug.csv"
CANONICAL_DEBUG_CSV = "canonical_batch_debug.csv"

# =========================
# Prompt
# Only first line changed as requested.
# =========================
SYSTEM_PROMPT = """You convert cleaned product-signature words into a single CANONICAL PRODUCT TYPE.

Rules:
- Output ONLY the product type as a short noun phrase (2-4 words), Title Case.
- Be specific where the text supports it: "Frozen Vannamei Shrimp", "Milled White Rice", "Cotton T-Shirt" — never just "Shrimp"/"Rice"/"Shirt".
- Push brand, model, size, colour, flavour, grade, weight, quantity, part/HS numbers to NOTHING — they are attributes, not the type.
- If the text is a document, generic ("general cargo"), or unintelligible, output exactly: Unknown
- No explanations, no punctuation, no quotes. One line only.

Examples:
"SAMSUNG GALAXY S26 512GB NEW OEM US SPEC BLUE" -> Smartphone
"1650 CARTONS FROZEN COOKED PEELED VANNAMEI SHRIMP" -> Frozen Vannamei Shrimp
"NEW SHANTUI SD22 BULLDOZER C/W ACCESSORIES" -> Bulldozer
"41600 BAGS PAKISTAN WHITE RICE 5PCT BROKENS DOUBLE POLISHED" -> Milled White Rice
"THIS SHIPMENT CONTAINS NO SOLID WOOD PACKING" -> Unknown
"""

# For grouped input, we add minimal user instruction at runtime so parsing is reliable.
# System prompt remains same except the first line above.
USER_GROUP_TEMPLATE = """Extract product types for each signature below.
Return exactly {n} lines.
Each line format must be: <number>. <Product Type>
No extra text.

Signatures:
{items}
"""

_CLEAN = re.compile(r'^[\s"\'`\-*.:]+|[\s"\'`\-*.:]+$')
_WS = re.compile(r"\s+")
_DATE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_NUM = re.compile(r"\d+([.,]\d+)?")
_CODE = re.compile(r"\b(?=[a-z0-9-]*\d)(?=[a-z0-9-]*[a-z])[a-z0-9-]{5,}\b", re.I)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

_STOP = {
    "the", "and", "for", "with", "not", "nes", "other", "others", "misc",
    "as", "per", "inv", "invoice", "no", "nos", "qty", "pcs", "pc", "set",
    "kg", "kgs", "gm", "gms", "mt", "ton", "tons", "ltr", "ml", "cm", "mm",
    "packed", "packing", "package", "packages", "carton", "cartons", "bag",
    "bags", "box", "boxes", "net", "gross", "weight", "wt", "hs", "code",
    "date", "dt", "po", "sb", "ref", "reference", "origin", "made", "in",
    "of", "to", "from", "x", "each", "total", "item", "items",
}

JUNK_MARKERS = {
    "general", "cargo", "merchandise", "documentation", "document",
    "shipment", "consolidated", "said", "contain", "sample",
}


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("_x000D_", " ").replace("\r", " ").replace("\n", " ")
    return _WS.sub(" ", s).strip()


def signature(s: str) -> str:
    s = normalize_text(s).lower()
    s = _DATE.sub(" ", s)
    s = _CODE.sub(" ", s)
    s = _NUM.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    toks = [t for t in s.split() if len(t) > 2 and t not in _STOP]
    return " ".join(sorted(set(toks)))


def is_probably_junk_from_sig(sig: str) -> bool:
    if len(sig) < 3:
        return True
    words = sig.split()
    if len(words) <= 2 and any(w in JUNK_MARKERS for w in words):
        return True
    return False


def postprocess_product_type(text: str) -> str:
    line = text.strip().splitlines()[0] if text and text.strip() else "Unknown"
    line = _CLEAN.sub("", line)
    if not line:
        return "Unknown"
    return line[:80]


def parse_numbered_response(raw: str, expected_n: int) -> Tuple[List[str], str]:
    """Parse numbered lines: 1. Product Type"""
    lines = [x.strip() for x in raw.strip().splitlines() if x.strip()]
    out: List[str] = []
    for line in lines:
        m = re.match(r"^\s*\d+\s*[\).:-]\s*(.+?)\s*$", line)
        if m:
            out.append(postprocess_product_type(m.group(1)))
        elif len(lines) == expected_n:
            out.append(postprocess_product_type(line))
    if len(out) == expected_n:
        return out, "ok"
    # fallback: if model returned JSON list of strings or objects
    try:
        data = json.loads(raw)
        tmp: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    tmp.append(postprocess_product_type(item))
                elif isinstance(item, dict):
                    tmp.append(postprocess_product_type(str(item.get("product_type", "Unknown"))))
        if len(tmp) == expected_n:
            return tmp, "ok_json_fallback"
    except Exception:
        pass
    return ["Unknown"] * expected_n, f"parse_failed_expected_{expected_n}_got_{len(out)}"


def call_gemma_for_signature_group(signatures: List[str]) -> Tuple[List[str], str, str, float]:
    items = "\n".join([f"{i+1}. {sig}" for i, sig in enumerate(signatures)])
    user_msg = USER_GROUP_TEMPLATE.format(n=len(signatures), items=items)
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{LLM_BASE_URL}/v1/chat/completions", json=body, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        raw = j["choices"][0]["message"]["content"]
        elapsed = time.time() - t0
        product_types, parse_status = parse_numbered_response(raw, len(signatures))
        return product_types, raw, parse_status, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return ["Unknown"] * len(signatures), str(e), "request_failed", elapsed


@dataclass
class Node:
    canonical_id: str
    canonical_name: str
    category: str


class OntologyMapper:
    def __init__(self, seed_csv: str, model_name: str, device: str, threshold: float, embed_batch_size: int):
        from sentence_transformers import SentenceTransformer
        self.threshold = threshold
        self.embed_batch_size = embed_batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.nodes: List[Node] = []
        self.alias_names: List[str] = []
        self.alias_node_idx: List[int] = []
        self.alias_matrix: Optional[np.ndarray] = None
        self.candidates: Dict[str, int] = {}
        self._load(seed_csv)

    def _load(self, seed_csv: str) -> None:
        names: List[str] = []
        node_idx: List[int] = []
        with open(seed_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"canonical_id", "canonical_name", "category"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Ontology CSV missing columns: {sorted(missing)}")
            for row in reader:
                node = Node(row["canonical_id"], row["canonical_name"], row["category"])
                self.nodes.append(node)
                aliases = [row["canonical_name"]]
                if row.get("aliases"):
                    aliases += [a.strip() for a in row["aliases"].split("|") if a.strip()]
                for a in aliases:
                    names.append(a)
                    node_idx.append(len(self.nodes) - 1)
        self.alias_names = names
        self.alias_node_idx = node_idx
        print(f"[{now_str()}] CANONICAL: embedding ontology aliases count={len(names)} nodes={len(self.nodes)} batch_size={self.embed_batch_size}")
        t0 = time.time()
        self.alias_matrix = self._encode(names)
        print(f"[{now_str()}] CANONICAL: ontology alias embeddings done elapsed={time.time()-t0:.2f}s")

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        arr = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.embed_batch_size,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)

    def map_batch(self, phrases: List[str], debug_writer, batch_id: int, total_batches: int) -> List[Tuple[str, str, str, float, str]]:
        t0 = time.time()
        clean_phrases = [p if p and p.strip().lower() not in ("unknown", "n/a") else "Unknown" for p in phrases]
        valid_idx = [i for i, p in enumerate(clean_phrases) if p != "Unknown"]
        results: List[Tuple[str, str, str, float, str]] = [("GEN.0000", "General / Unknown", "Generic", 0.0, "unknown") for _ in phrases]
        if valid_idx:
            valid_phrases = [clean_phrases[i] for i in valid_idx]
            qmat = self._encode(valid_phrases)
            sims = qmat @ self.alias_matrix.T  # type: ignore[union-attr]
            best_j = np.argmax(sims, axis=1)
            best_s = np.max(sims, axis=1)
            for local_i, original_i in enumerate(valid_idx):
                sim = float(best_s[local_i])
                alias_j = int(best_j[local_i])
                phrase = clean_phrases[original_i]
                if sim >= self.threshold:
                    node = self.nodes[self.alias_node_idx[alias_j]]
                    results[original_i] = (node.canonical_id, node.canonical_name, node.category, round(sim, 3), "mapped")
                else:
                    self.candidates[phrase] = self.candidates.get(phrase, 0) + 1
                    results[original_i] = ("GEN.0000", "General / Unknown", "Generic", round(sim, 3), "unmapped")
        elapsed = time.time() - t0
        preview = " | ".join(clean_phrases[:5])[:DEBUG_SIGNATURE_PREVIEW_CHARS]
        debug_writer.writerow({
            "timestamp": now_str(),
            "canonical_batch_id": batch_id,
            "canonical_total_batches": total_batches,
            "product_type_count": len(phrases),
            "product_types_preview": preview,
            "elapsed_sec": round(elapsed, 3),
        })
        print(f"[{now_str()}] CANONICAL batch {batch_id}/{total_batches} product_types={len(phrases)} elapsed={elapsed:.2f}s preview={preview}")
        return results


def chunked(xs: List[str], n: int):
    for i in range(0, len(xs), n):
        yield i // n + 1, xs[i:i+n]


def main() -> None:
    run_start = time.time()
    ensure_dir(OUTPUT_DIR)
    print("=" * 90)
    print(f"[{now_str()}] PRODUCT TYPE SIGNATURE RUN START")
    print("=" * 90)
    print(f"INPUT_FILE={INPUT_FILE}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    print(f"PRODUCT_DESC_COL={PRODUCT_DESC_COL}")
    print(f"LLM_BASE_URL={LLM_BASE_URL}")
    print(f"LLM_MODEL={LLM_MODEL}")
    print(f"SIGNATURES_PER_LLM_CALL={SIGNATURES_PER_LLM_CALL}")
    print(f"PARALLEL_WORKERS={PARALLEL_WORKERS}")
    print(f"EMBED_MODEL={EMBED_MODEL} device={EMBED_DEVICE} batch={EMBED_BATCH_SIZE}")
    print("=" * 90)

    t0 = time.time()
    df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)
    if PRODUCT_DESC_COL not in df.columns:
        raise ValueError(f"Missing column '{PRODUCT_DESC_COL}'. Available columns: {list(df.columns)}")
    df = df[[PRODUCT_DESC_COL]].copy()
    df.insert(0, "record_id", np.arange(1, len(df) + 1))
    print(f"[{now_str()}] READ input rows={len(df)} elapsed={time.time()-t0:.2f}s")

    t0 = time.time()
    df["signature"] = df[PRODUCT_DESC_COL].map(signature)
    df["is_junk"] = df["signature"].map(is_probably_junk_from_sig)
    raw_rows = len(df)
    junk_rows = int(df["is_junk"].sum())
    nonjunk = df[~df["is_junk"]].copy()
    unique_signatures = sorted(nonjunk["signature"].drop_duplicates().tolist())
    print(f"[{now_str()}] SIGNATURES created elapsed={time.time()-t0:.2f}s")
    print(f"[{now_str()}] RAW rows={raw_rows} junk_rows={junk_rows} nonjunk_rows={len(nonjunk)} unique_nonjunk_signatures={len(unique_signatures)}")
    print(f"[{now_str()}] LLM total signature-groups={len(unique_signatures)} total_llm_batches={(len(unique_signatures)+SIGNATURES_PER_LLM_CALL-1)//SIGNATURES_PER_LLM_CALL}")

    sig_to_product_type: Dict[str, str] = {}
    llm_batches = list(chunked(unique_signatures, SIGNATURES_PER_LLM_CALL))
    total_llm_batches = len(llm_batches)

    llm_debug_path = os.path.join(OUTPUT_DIR, LLM_DEBUG_CSV)
    with open(llm_debug_path, "w", newline="", encoding="utf-8") as fdebug:
        fieldnames = ["timestamp", "llm_batch_id", "llm_total_batches", "signature_count", "signatures", "product_types", "parse_status", "elapsed_sec", "raw_response"]
        writer = csv.DictWriter(fdebug, fieldnames=fieldnames)
        writer.writeheader()

        def process_llm_batch(item):
            batch_id, sigs = item
            preview = " | ".join(sigs)[:DEBUG_SIGNATURE_PREVIEW_CHARS]
            print(f"[{now_str()}] PRODUCT_TYPE batch {batch_id}/{total_llm_batches} signatures={len(sigs)} running: {preview}")
            product_types, raw, parse_status, elapsed = call_gemma_for_signature_group(sigs)
            return batch_id, sigs, product_types, raw, parse_status, elapsed

        with cf.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
            futs = [ex.submit(process_llm_batch, b) for b in llm_batches]
            completed = 0
            for fut in cf.as_completed(futs):
                batch_id, sigs, product_types, raw, parse_status, elapsed = fut.result()
                completed += 1
                for sig, pt in zip(sigs, product_types):
                    sig_to_product_type[sig] = pt
                writer.writerow({
                    "timestamp": now_str(),
                    "llm_batch_id": batch_id,
                    "llm_total_batches": total_llm_batches,
                    "signature_count": len(sigs),
                    "signatures": json.dumps(sigs, ensure_ascii=False),
                    "product_types": json.dumps(product_types, ensure_ascii=False),
                    "parse_status": parse_status,
                    "elapsed_sec": round(elapsed, 3),
                    "raw_response": raw[:2000],
                })
                fdebug.flush()
                done_sigs = len(sig_to_product_type)
                print(f"[{now_str()}] PRODUCT_TYPE completed batch {batch_id}/{total_llm_batches} overall_batches={completed}/{total_llm_batches} signatures_done={done_sigs}/{len(unique_signatures)} status={parse_status} elapsed={elapsed:.2f}s")

    # Canonical mapping in batches of product types
    print(f"[{now_str()}] CANONICAL mapping start unique_product_types={len(set(sig_to_product_type.values()))}")
    mapper = OntologyMapper(ONTOLOGY_SEED_CSV, EMBED_MODEL, EMBED_DEVICE, ONTOLOGY_THRESHOLD, EMBED_BATCH_SIZE)
    sigs_for_canonical = list(sig_to_product_type.keys())
    product_types_for_canonical = [sig_to_product_type[s] for s in sigs_for_canonical]
    canonical_results: Dict[str, Tuple[str, str, str, float, str]] = {}

    canonical_debug_path = os.path.join(OUTPUT_DIR, CANONICAL_DEBUG_CSV)
    canonical_total_batches = (len(product_types_for_canonical) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    with open(canonical_debug_path, "w", newline="", encoding="utf-8") as fcdebug:
        cwriter = csv.DictWriter(fcdebug, fieldnames=["timestamp", "canonical_batch_id", "canonical_total_batches", "product_type_count", "product_types_preview", "elapsed_sec"])
        cwriter.writeheader()
        for batch_id, start in enumerate(range(0, len(product_types_for_canonical), EMBED_BATCH_SIZE), start=1):
            pts = product_types_for_canonical[start:start+EMBED_BATCH_SIZE]
            sigs_batch = sigs_for_canonical[start:start+EMBED_BATCH_SIZE]
            mapped = mapper.map_batch(pts, cwriter, batch_id, canonical_total_batches)
            for sig, res in zip(sigs_batch, mapped):
                canonical_results[sig] = res
            fcdebug.flush()

    # Build output rows
    print(f"[{now_str()}] WRITING outputs")
    out_rows = []
    for _, row in df.iterrows():
        sig = row["signature"]
        if row["is_junk"]:
            llm_pt = "Unknown"
            cid, cname, cat, sim, status = "GEN.0000", "General / Unknown", "Generic", 1.0, "junk-filter"
        else:
            llm_pt = sig_to_product_type.get(sig, "Unknown")
            cid, cname, cat, sim, status = canonical_results.get(sig, ("GEN.0000", "General / Unknown", "Generic", 0.0, "missing"))
        out_rows.append({
            "record_id": int(row["record_id"]),
            "product_description": row[PRODUCT_DESC_COL],
            "signature": sig,
            "llm_product_type": llm_pt,
            "canonical_id": cid,
            "canonical_product_type": cname,
            "canonical_category": cat,
            "similarity": sim,
            "status": status,
        })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV), index=False)

    sig_map_rows = []
    for sig in sigs_for_canonical:
        cid, cname, cat, sim, status = canonical_results[sig]
        sig_map_rows.append({
            "signature": sig,
            "llm_product_type": sig_to_product_type[sig],
            "canonical_id": cid,
            "canonical_product_type": cname,
            "canonical_category": cat,
            "similarity": sim,
            "status": status,
            "raw_row_count": int((df["signature"] == sig).sum()),
        })
    pd.DataFrame(sig_map_rows).to_csv(os.path.join(OUTPUT_DIR, SIGNATURE_MAP_CSV), index=False)

    cand_rows = [{"product_type": k, "count": v} for k, v in sorted(mapper.candidates.items(), key=lambda x: -x[1])]
    pd.DataFrame(cand_rows).to_csv(os.path.join(OUTPUT_DIR, UNMAPPED_CANDIDATES_CSV), index=False)

    elapsed_total = time.time() - run_start
    summary = {
        "started_at": None,
        "finished_at": now_str(),
        "elapsed_sec": round(elapsed_total, 3),
        "elapsed_min": round(elapsed_total / 60, 3),
        "raw_rows": raw_rows,
        "junk_rows": junk_rows,
        "nonjunk_rows": len(nonjunk),
        "unique_nonjunk_signatures": len(unique_signatures),
        "llm_batches": total_llm_batches,
        "signatures_per_llm_call": SIGNATURES_PER_LLM_CALL,
        "parallel_workers": PARALLEL_WORKERS,
        "embed_batch_size": EMBED_BATCH_SIZE,
        "mapped_signatures": sum(1 for x in canonical_results.values() if x[4] == "mapped"),
        "unmapped_signatures": sum(1 for x in canonical_results.values() if x[4] == "unmapped"),
        "output_csv": OUTPUT_CSV,
        "signature_map_csv": SIGNATURE_MAP_CSV,
        "llm_debug_csv": LLM_DEBUG_CSV,
        "canonical_debug_csv": CANONICAL_DEBUG_CSV,
    }
    with open(os.path.join(OUTPUT_DIR, RUN_SUMMARY_JSON), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print(f"[{now_str()}] RUN COMPLETE elapsed={elapsed_total/60:.2f} min")
    print(f"Output: {os.path.join(OUTPUT_DIR, OUTPUT_CSV)}")
    print(f"Signature map: {os.path.join(OUTPUT_DIR, SIGNATURE_MAP_CSV)}")
    print(f"LLM debug: {llm_debug_path}")
    print(f"Canonical debug: {canonical_debug_path}")
    print(f"Summary: {os.path.join(OUTPUT_DIR, RUN_SUMMARY_JSON)}")
    print("=" * 90)


if __name__ == "__main__":
    main()
