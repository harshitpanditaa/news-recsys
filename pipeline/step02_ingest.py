"""STEP 2 - Collect & Ingest Data.

Source 1 (files / CSV-TSV): MIND-small, Microsoft News Dataset
    news.tsv       article catalogue (id, category, title, abstract, entities)
    behaviors.tsv  impression logs (user, time, click history, shown articles + clicked flag)
Source 2 (REST API): The Guardian Open Platform content API
    live, recently published articles (title, section, trail text, date)

Raw files are stored untouched in the LANDING zone together with a manifest
(file name, size, MD5 checksum, URL, timestamp) so every later layer can be
traced back to its source.
"""
import glob
import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import timedelta

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    GUARDIAN_API_KEY, GUARDIAN_DAYS_BACK, GUARDIAN_MAX_PAGES, GUARDIAN_URL,
    LANDING, MIND_MIRRORS, MIND_SPLITS, banner, now_utc, write_json,
)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
                done += len(block)
        if total and done != total:
            raise IOError(f"Incomplete download: {done} of {total} bytes")
    os.replace(tmp, dest)


def ingest_mind():
    """Download + unzip MIND-small train and dev splits (skips if already present)."""
    manifest = []
    for split, zip_name in MIND_SPLITS.items():
        split_dir = os.path.join(LANDING, "mind", split)
        os.makedirs(split_dir, exist_ok=True)
        zip_path = os.path.join(LANDING, "mind", zip_name)
        url = None

        if not os.path.exists(os.path.join(split_dir, "behaviors.tsv")):
            if os.path.exists(zip_path) and not zipfile.is_zipfile(zip_path):
                os.remove(zip_path)  # leftover from a failed download
            if os.path.exists(zip_path):
                url = "(provided manually)"
            errors = []
            for base in ([] if url else MIND_MIRRORS):  # try each mirror in turn
                candidate = f"{base}/{zip_name}"
                print(f"Downloading {candidate} ...")
                try:
                    download(candidate, zip_path)
                    if not zipfile.is_zipfile(zip_path):
                        raise IOError("downloaded file is not a valid zip")
                    url = candidate
                    break
                except Exception as e:
                    errors.append(f"{candidate}: {e}")
                    print(f"  failed ({e}); trying next mirror")
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
            if url is None:
                raise SystemExit(
                    "Could not download MIND from any mirror:\n  " + "\n  ".join(errors) +
                    f"\nManual fallback: download {zip_name} from "
                    f"https://huggingface.co/datasets/Recommenders/MIND and place it at {zip_path}, "
                    "then re-run this step."
                )
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(split_dir)
            # some copies of the zip wrap the files in a sub-folder: move them up
            if not os.path.exists(os.path.join(split_dir, "behaviors.tsv")):
                for root, _, files in os.walk(split_dir):
                    for f in files:
                        if f.endswith((".tsv", ".vec")):
                            os.replace(os.path.join(root, f), os.path.join(split_dir, f))
            print(f"  OK: {zip_name} from {url}")
        else:
            print(f"MIND {split}: already in landing zone, skipping download")

        for fname in ("news.tsv", "behaviors.tsv"):
            fpath = os.path.join(split_dir, fname)
            with open(fpath, "rb") as f:
                n_lines = sum(1 for _ in f)
            manifest.append({
                "source": "MIND-small", "source_type": "file (TSV)", "split": split,
                "file": fpath, "source_url": url, "bytes": os.path.getsize(fpath),
                "rows": n_lines, "md5": md5(fpath), "ingested_at_utc": now_utc().isoformat(),
            })
            print(f"  {split}/{fname}: {n_lines:,} rows, {os.path.getsize(fpath)/1e6:.1f} MB")
    return manifest


def ingest_guardian():
    """Pull recently published Guardian articles page by page; each raw JSON page is saved."""
    run_id = now_utc().strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(LANDING, "guardian", f"run={run_id}")
    os.makedirs(run_dir, exist_ok=True)
    from_date = (now_utc() - timedelta(days=GUARDIAN_DAYS_BACK)).strftime("%Y-%m-%d")
    params = {
        "api-key": GUARDIAN_API_KEY, "from-date": from_date, "order-by": "newest",
        "page-size": 50, "show-fields": "trailText,wordcount,thumbnail,byline",
    }
    if GUARDIAN_API_KEY == "test":
        print("NOTE: using the Guardian public 'test' key. Register a free key at "
              "https://open-platform.theguardian.com/access/ and set GUARDIAN_API_KEY.")

    pages, n_articles, page, total_pages = [], 0, 1, 1
    while page <= min(total_pages, GUARDIAN_MAX_PAGES):
        for attempt in range(4):
            r = requests.get(GUARDIAN_URL, params={**params, "page": page}, timeout=30)
            if r.status_code == 429:  # rate limited -> back off and retry
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        payload = r.json()
        resp = payload["response"]
        total_pages = resp.get("pages", 1)
        out = os.path.join(run_dir, f"page_{page:03d}.json")
        with open(out, "w") as f:
            json.dump(payload, f)
        n = len(resp.get("results", []))
        n_articles += n
        pages.append({"page": page, "file": out, "articles": n, "md5": md5(out)})
        print(f"  Guardian page {page}/{min(total_pages, GUARDIAN_MAX_PAGES)}: {n} articles")
        page += 1
        time.sleep(1.1)  # free tier allows ~1 call/second

    return [{
        "source": "The Guardian Open Platform", "source_type": "REST API (JSON)",
        "run_id": run_id, "endpoint": GUARDIAN_URL, "from_date": from_date,
        "pages": pages, "articles": n_articles, "ingested_at_utc": now_utc().isoformat(),
    }]


def run():
    banner("STEP 2: INGEST - MIND files + Guardian API -> landing zone")
    manifest = {"mind": ingest_mind()}
    try:
        manifest["guardian"] = ingest_guardian()
    except Exception as e:
        # The pipeline can still run on MIND alone; earlier Guardian runs remain usable.
        print(f"WARNING: Guardian API ingestion failed ({e}).")
        manifest["guardian"] = []
        if not glob.glob(os.path.join(LANDING, "guardian", "run=*", "*.json")):
            print("No Guardian data in landing zone; fresh-article recommendations will be skipped.")
    path = os.path.join(LANDING, "_manifests", f"ingest_{now_utc().strftime('%Y%m%dT%H%M%SZ')}.json")
    write_json(manifest, path)
    print(f"\nIngestion manifest written to {path}")
    return manifest


if __name__ == "__main__":
    run()
