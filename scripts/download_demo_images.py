#!/usr/bin/env python3
"""
Download only the images needed by demo_data JSONL files from the OpenScene
dataset on Hugging Face.

Strategy:
  1. Scan demo_data JSONL files → collect image paths → extract session IDs.
  2. Look up which tgz shard each session belongs to (via index JSON).
  3. Download the full shard tgz and extract only the needed files, then delete the tgz.
"""

import json
import os
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
# 使用 Hugging Face 镜像（只需替换 endpoint，URL 路径保持一致）
HF_ENDPOINT = "https://hf-mirror.com"
HF_BASE = f"{HF_ENDPOINT}/datasets/OpenDriveLab/OpenScene/resolve/main"
INDEX_URL = f"{HF_BASE}/openscene-v1.1/openscene_sensor_trainval_0-199.json"

OUTPUT_ROOT = Path("data") / "navsim_v1.1_all" / "dataset"

DEMO_FILES = [
    "demo_data/navsim/navsim_answer_demo100.jsonl",
    "demo_data/navsim/navsim_cot_demo100.jsonl",
    "demo_data/navsim/navsim_vis4_text2_demo100.jsonl",
]

# ────────────────────────────────────────────────────────────────


def http_get(url: str, timeout: int = 60) -> bytes:
    """Download with manual 308 redirect handling (hf-mirror returns 308 → huggingface.co)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 308:
            new_url = e.headers.get("Location")
            if new_url:
                req2 = urllib.request.Request(new_url, headers={"User-Agent": "Mozilla/5.0"})
                resp2 = urllib.request.urlopen(req2, timeout=timeout)
                return resp2.read()
        raise


def http_stream(url: str):
    """Streaming download with manual 308 redirect handling."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        return resp
    except urllib.error.HTTPError as e:
        if e.code == 308:
            new_url = e.headers.get("Location")
            if new_url:
                req2 = urllib.request.Request(new_url, headers={"User-Agent": "Mozilla/5.0"})
                return urllib.request.urlopen(req2, timeout=600)
        raise


def collect_needed_images(demo_files: list[str]) -> dict[str, list[dict]]:
    needed: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    for fpath in demo_files:
        if not os.path.exists(fpath):
            print(f"  [WARN] File not found: {fpath}")
            continue
        with open(fpath) as f:
            for line in f:
                data = json.loads(line)
                for img_path in data.get("images", []):
                    parts = img_path.split("/")
                    if len(parts) < 7:
                        continue
                    session_id = parts[4]
                    filename = parts[6]
                    key = (session_id, filename)
                    if key not in seen:
                        seen.add(key)
                        needed[session_id].append({
                            "session_id": session_id,
                            "cam": parts[5],
                            "filename": filename,
                        })

    total = sum(len(v) for v in needed.values())
    print(f"Found {len(needed)} unique sessions with {total} total images")
    return dict(needed)


def load_shard_index() -> dict[str, list[str]]:
    data = http_get(INDEX_URL)
    return json.loads(data)


def build_shard_map(index: dict[str, list[str]]) -> dict[str, set[str]]:
    session_to_shards: dict[str, set[str]] = defaultdict(set)
    for shard_name, sessions in index.items():
        for sess in sessions:
            session_to_shards[sess].add(shard_name)
    return dict(session_to_shards)


def tgz_url(shard_name: str) -> str:
    shard_num = shard_name.rsplit("_", 1)[-1]
    return f"{HF_BASE}/openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{shard_num}.tgz"


def download_and_extract_selective(
    shard_name: str,
    needed_sessions: set[str],
    needed_files: dict[str, list[dict]],
    output_root: Path,
) -> int:
    url = tgz_url(shard_name)
    tgz_filename = url.rsplit("/", 1)[-1]
    local_tgz = Path(".claude_tmp") / tgz_filename
    local_tgz.parent.mkdir(parents=True, exist_ok=True)

    # Build set of internal tgz paths we need
    # tgz internal: openscene-v1.1/sensor_blobs/trainval/{session}/{cam}/{file}
    want_internal: set[str] = set()
    for sess in needed_sessions:
        for entry in needed_files.get(sess, []):
            want_internal.add(
                f"openscene-v1.1/sensor_blobs/trainval/{entry['session_id']}/{entry['cam']}/{entry['filename']}"
            )

    try:
        print(f"  Downloading {tgz_filename} (~4.7GB)...", flush=True)
        resp = http_stream(url)
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 16 * 1024 * 1024

        with open(local_tgz, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / total_size * 100 if total_size else 0
                print(f"\r  Downloading: {downloaded/1024/1024:.0f}/{total_size/1024/1024:.0f} MB ({pct:.0f}%)", end="", flush=True)
        print()

        print(f"  Extracting needed files ...", flush=True)
        extracted_count = 0
        with tarfile.open(local_tgz, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if member.name in want_internal:
                    parts = Path(member.name).parts
                    if len(parts) >= 6:
                        sess, cam, fname = parts[3], parts[4], parts[5]
                        out_path = output_root / "sensor_blobs" / "trainval" / sess / cam / fname
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        src_f = tar.extractfile(member)
                        if src_f is not None:
                            with open(out_path, "wb") as dst_f:
                                dst_f.write(src_f.read())
                            extracted_count += 1
                            print(f"    ✓ {sess}/{cam}/{fname}")

        print(f"  Extracted {extracted_count} files from {tgz_filename}")
        return extracted_count

    except Exception as e:
        print(f"  [ERROR] {shard_name}: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        if local_tgz.exists():
            local_tgz.unlink()
            print(f"  Cleaned up {tgz_filename}")


def main():
    print("=" * 60)
    print("NavSim Demo Data Image Downloader")
    print("=" * 60)
    print()

    print("[1/4] Scanning demo data files ...")
    needed_files = collect_needed_images(DEMO_FILES)
    total_images = sum(len(v) for v in needed_files.values())

    print("\n[2/4] Loading shard index ...")
    index = load_shard_index()
    session_to_shards = build_shard_map(index)

    print("\n[3/4] Determining required shards ...")
    shard_to_sessions: dict[str, set[str]] = defaultdict(set)
    for session_id in needed_files:
        shards = session_to_shards.get(session_id, set())
        if not shards:
            print(f"  [WARN] Session '{session_id}' not found in any shard!")
        for s in shards:
            shard_to_sessions[s].add(session_id)

    for shard_name in sorted(shard_to_sessions.keys()):
        n_sessions = len(shard_to_sessions[shard_name])
        n_files = sum(len(needed_files[s]) for s in shard_to_sessions[shard_name])
        print(f"    {shard_name}: {n_sessions} session(s), {n_files} file(s)")

    total_download_gb = len(shard_to_sessions) * 4.7
    print(f"\n  Total: {len(shard_to_sessions)} shard(s), ~{total_download_gb:.0f} GB download")
    print(f"  To extract {total_images} images to: {OUTPUT_ROOT.resolve()}")

    # ── LIMIT: only first 2 shards for testing ──
    # Remove this block to process all shards
    limited_shards = dict(sorted(shard_to_sessions.items())[:2])
    print(f"\n  [TEST MODE] Limiting to 2 shards: {list(limited_shards.keys())}")
    shard_to_sessions = limited_shards
    # ─────────────────────────────────────────────

    try:
        reply = input("\nContinue? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return

    print("\n[4/4] Downloading and extracting ...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    total_extracted = 0
    for shard_name in sorted(shard_to_sessions.keys()):
        print(f"\n--- Processing {shard_name} ---")
        count = download_and_extract_selective(
            shard_name, shard_to_sessions[shard_name], needed_files, OUTPUT_ROOT
        )
        total_extracted += count

    print(f"\n{'=' * 60}")
    print(f"Done! Extracted {total_extracted}/{total_images} images")
    print(f"Output directory: {OUTPUT_ROOT.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()