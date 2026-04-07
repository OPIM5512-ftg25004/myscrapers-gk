# main.py
# Build a single, ever-growing CSV from all structured JSONL files.
# Reads:  gs://<bucket>/<STRUCTURED_PREFIX>/run_id=*/jsonl/*.jsonl
# Writes: gs://<bucket>/<STRUCTURED_PREFIX>/datasets/listings_master.csv  (atomic publish)


import csv
import io
import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage

# -------------------- ENV --------------------
BUCKET_NAME        = os.getenv("GCS_BUCKET")                      # REQUIRED
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured") # e.g., "structured"
OUTPUT_FILENAME = "listings_master_llm.csv" # Changed to reflect the listings_master_llm.csv destination file

storage_client = storage.Client()

# Accept BOTH runIDs:
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")        # 20251026170002

# Stable CSV schema for students
CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",
    "price", "year", "make", "model", "mileage",
    "transmission", "condition", "color", "city",
    "state", "zip_code", "source_txt", "llm_model"
]

def _list_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:  # populate it.prefixes
        pass
    run_ids = []
    for p in getattr(it, "prefixes", []):
        tail = p.rstrip("/").split("/")[-1]           # e.g. run_id=20251026170002
        if tail.startswith("run_id="):
            rid = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(rid) or RUN_ID_PLAIN_RE.match(rid):
                run_ids.append(rid)
    return sorted(run_ids)

def _jsonl_records_for_run(bucket: str, structured_prefix: str, run_id: str):
    """Yield dict records from .jsonl under .../run_id=<run_id>/jsonl_llm/ (one JSON per file)."""
    b = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/jsonl_llm/"
    for blob in b.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        data = blob.download_as_text()
        line = data.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            # ensure required keys exist
            rec.setdefault("run_id", run_id)
            yield rec
        except Exception:
            continue

def _run_id_to_dt(rid: str) -> datetime:
    if RUN_ID_ISO_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if RUN_ID_PLAIN_RE.match(rid):
        return datetime.strptime(rid, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

def _open_gcs_text_writer(bucket: str, key: str):
    """Open a text-mode writer to GCS."""
    b = storage_client.bucket(bucket)
    blob = b.blob(key)
    return blob.open("w")  # newline handled by csv module

def _write_csv(records: Iterable[Dict], dest_key: str, columns=CSV_COLUMNS) -> int:
    n = 0
    with _open_gcs_text_writer(BUCKET_NAME, dest_key) as out:
        # Ensures that the old files do not break from this function with restval=""
        w = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore", restval="")
        w.writeheader()
        for rec in records:
            zip_val = rec.get("zip_code")
            if zip_val:
                # Convert to string, strip any decimal points (like '1234.0'), and pad to 5 digits
                zip_str = str(zip_val).split('.')[0].strip()
                if zip_str.isdigit():
                    rec["zip_code"] = zip_str.zfill(5)
            row = {c: rec.get(c, None) for c in columns}
            w.writerow(row)
            n += 1
    return n

### Assignment 3 Part
# Look in the 'jsonl_llm' sub-directory pointing towards the sub-folder as created by the LLM extractor
def _llm_jsonl_records_for_run(bucket: str, structured_prefix: str, run_id: str):
    b = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/jsonl_llm/" 
    for blob in b.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        try:
            rec = json.loads(blob.download_as_text().strip())
            yield rec
        except Exception:
            continue

def _get_existing_master_data(bucket_name: str, key: str) -> Dict[str, Dict]:
    """Downloads existing master CSV and returns a dict keyed by post_id."""
    b = storage_client.bucket(bucket_name)
    blob = b.blob(key)
    data = {}
    if not blob.exists():
        return data
    
    try:
        content = blob.download_as_text()
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            pid = row.get("post_id")
            if pid:
                data[pid] = row
    except Exception:
        pass 
    return data

# Add the new HTTP entry point for the materialize-http function
def materialize_http(request: Request):
    """
    Optimized Materialize:
    1. Loads existing Master CSV.
    2. Only scans GCS folders from the last 75 minutes.
    3. Merges/Deduplicates new data into the Master records.
    """
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

        # 1. Load the EXISTING 'Master' data first (The "Past Code" approach)
        # This prevents recreating the entire dataset from scratch every hour.
        final_key = f"{STRUCTURED_PREFIX}/datasets/{OUTPUT_FILENAME}"
        master_records = _get_existing_master_data(BUCKET_NAME, final_key)
        
        # 2. Filter for ONLY recent runs (The "75-minute" performance fix)
        all_run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        limit_time = datetime.now(timezone.utc) - timedelta(minutes=75)
        
        # We only care about runs newer than 75 mins ago
        recent_runs = [r for r in all_run_ids if _run_id_to_dt(r) > limit_time]

        if not recent_runs:
            return jsonify({
                "ok": True, 
                "message": "No new runs found in the last 75 minutes. Master CSV is already up to date.",
                "total_listings": len(master_records)
            }), 200

        # 3. Fetch ONLY the new records and merge/deduplicate
        for rid in recent_runs:
            # Use your LLM-specific fetcher
            for rec in _llm_jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid):
                pid = rec.get("post_id")
                if not pid: 
                    continue
                
                # Update the record if it's new OR if this specific run is newer 
                # than what we currently have in our master dictionary
                prev = master_records.get(pid)
                if (prev is None) or (_run_id_to_dt(rid) >= _run_id_to_dt(prev.get("run_id", ""))):
                    # Attach the run_id to the record so we can track freshness
                    rec["run_id"] = rid 
                    master_records[pid] = rec

        # 4. Save the merged result back to GCS
        rows_written = _write_csv(master_records.values(), final_key, columns=CSV_COLUMNS)

        return jsonify({
            "ok": True,
            "recent_runs_scanned": len(recent_runs),
            "total_listings_in_master": rows_written,
            "output_csv": f"gs://{BUCKET_NAME}/{final_key}"
        }), 200

    except Exception as e:
        # Improved error logging for debugging
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {str(e)}"}), 500