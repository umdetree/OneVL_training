#!/usr/bin/env python3
"""
NavSim 数据集下载 + prepare 脚本

从 OpenScene Hugging Face 数据集下载 session 图像，扫描本地目录，
生成 raw_index.jsonl（路径三元组）和 images/ 软链接。

用法:
    # 下载约 20 个 session（~3 个 shard），每 session 最多 10 组三元组
    python -m scripts.prepare.navsim \\
        --navsim_root /root/autodl-tmp/OneVL_training/navsim_v1.1_all \\
        --output_dir /root/autodl-tmp/OneVL_training/data/navsim \\
        --num_sessions 20 --max_samples_per_session 10

    # 只扫描本地已有数据，不下载
    python -m scripts.prepare.navsim \\
        --navsim_root /root/autodl-tmp/OneVL_training/navsim_v1.1_all \\
        --output_dir /root/autodl-tmp/OneVL_training/data/navsim \\
        --scan_only

数据流:
    NavSim 原始目录                    data/navsim/
    ├── dataset/                       ├── images/          ← 软链接
    │   └── sensor_blobs/              │   └── 2021.06.../
    │       └── trainval/              │       └── CAM_F0/
    │           └── 2021.06.../        │           └── xxx.jpg
    │               └── CAM_F0/        ├── raw_index.jsonl  ← 输出
    │                   └── xxx.jpg    └── ...
    └── ...

Session 命名格式:
    2021.05.12.19.36.12_veh-35_00005_00204
    └── date/time ──┘ └─ veh ─┘ └─ frames ─┘
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── Hugging Face 镜像配置 ──────────────────────────────────────────────────
HF_ENDPOINT = "https://hf-mirror.com"
HF_BASE = f"{HF_ENDPOINT}/datasets/OpenDriveLab/OpenScene/resolve/main"
INDEX_URL = f"{HF_BASE}/openscene-v1.1/openscene_sensor_trainval_0-199.json"


# ═══════════════════════════════════════════════════════════════════════════
# 下载工具
# ═══════════════════════════════════════════════════════════════════════════


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


def load_shard_index() -> dict[str, list[str]]:
    """Load shard → session_ids mapping from HF index JSON."""
    print(f"  Loading shard index from HF...")
    data = http_get(INDEX_URL)
    return json.loads(data)


def build_shard_map(index: dict[str, list[str]]) -> dict[str, set[str]]:
    """Build session_id → shard_names reverse mapping."""
    session_to_shards: dict[str, set[str]] = defaultdict(set)
    for shard_name, sessions in index.items():
        for sess in sessions:
            session_to_shards[sess].add(shard_name)
    return dict(session_to_shards)


def tgz_url(shard_name: str) -> str:
    shard_num = shard_name.rsplit("_", 1)[-1]
    return f"{HF_BASE}/openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{shard_num}.tgz"


def _download_file(url: str, dest: Path) -> None:
    """Stream a URL to a local file with progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = http_stream(url)
    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    chunk = 16 * 1024 * 1024

    with open(dest, "wb") as f:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            pct = downloaded / total * 100 if total else 0
            print(f"\r  Downloading: {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB ({pct:.0f}%)", end="", flush=True)
    print()


def _extract_session_files(
    tar: tarfile.TarFile,
    session_id: str,
    navsim_root: Path,
    max_frames: int = 0,
) -> int:
    """Extract CAM_F0 images for a single session from tar.

    Internal path format:
      openscene-v1.1/sensor_blobs/trainval/{session_id}/CAM_F0/{filename}.jpg

    Returns number of files extracted.
    """
    prefix = f"openscene-v1.1/sensor_blobs/trainval/{session_id}/CAM_F0/"
    extracted = 0

    for member in tar.getmembers():
        if not member.isfile():
            continue
        if not member.name.startswith(prefix):
            continue
        if not member.name.endswith(".jpg"):
            continue

        # Internal: openscene-v1.1/sensor_blobs/trainval/{sess}/CAM_F0/{fname}
        parts = Path(member.name).parts
        fname = parts[-1]  # filename

        out = navsim_root / "dataset" / "sensor_blobs" / "trainval" / session_id / "CAM_F0" / fname
        if out.exists():
            extracted += 1
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        src = tar.extractfile(member)
        if src is None:
            continue
        with open(out, "wb") as dst:
            dst.write(src.read())
        extracted += 1

        if max_frames > 0 and extracted >= max_frames:
            break

    return extracted


def download_and_extract_shard(
    shard_name: str,
    session_ids: list[str],
    navsim_root: Path,
    max_frames: int = 50,
) -> dict[str, int]:
    """Download one shard tgz and extract CAM_F0 images for all listed sessions.

    Returns dict mapping session_id → extracted file count.
    """
    # 检查哪些 session 还需要下载
    pending: list[str] = []
    results: dict[str, int] = {}
    for sess in session_ids:
        session_dir = navsim_root / "dataset" / "sensor_blobs" / "trainval" / sess / "CAM_F0"
        if session_dir.is_dir():
            existing = len(list(session_dir.glob("*.jpg")))
            if existing >= max_frames:
                print(f"  ✓ {sess}: already have {existing} frames, skipping")
                results[sess] = existing
                continue
        pending.append(sess)

    if not pending:
        print(f"  All sessions in {shard_name} already have enough frames")
        return results

    url = tgz_url(shard_name)
    tgz_name = url.rsplit("/", 1)[-1]
    local_tgz = Path(".claude_tmp") / tgz_name

    try:
        if not local_tgz.exists():
            print(f"  Downloading {tgz_name} (~4.7GB) ...", flush=True)
            _download_file(url, local_tgz)
        else:
            print(f"  Using cached {tgz_name} ...", flush=True)

        print(f"  Extracting {len(pending)} session(s) from {tgz_name} ...", flush=True)
        with tarfile.open(local_tgz, "r:gz") as tar:
            for sess in pending:
                count = _extract_session_files(tar, sess, navsim_root, max_frames)
                results[sess] = count
                if count > 0:
                    print(f"    ✓ {sess}: {count} files")
                else:
                    print(f"    ✗ {sess}: no files found")

        return results

    except Exception as e:
        print(f"  [ERROR] processing {shard_name}: {e}")
        import traceback
        traceback.print_exc()
        for sess in pending:
            if sess not in results:
                results[sess] = 0
        return results
    finally:
        if local_tgz.exists():
            local_tgz.unlink()
            print(f"  Cleaned up {tgz_name}")


# ═══════════════════════════════════════════════════════════════════════════
# 本地扫描 + 滑动窗口
# ═══════════════════════════════════════════════════════════════════════════


def scan_local_sessions(
    navsim_root: Path,
    min_frames: int = 3,
) -> dict[str, list[Path]]:
    """扫描本地 navsim_root，按 session 分组返回帧列表。

    只返回帧数 >= min_frames 的 session（否则无法形成三元组）。
    """
    trainval = navsim_root / "dataset" / "sensor_blobs" / "trainval"
    if not trainval.is_dir():
        print(f"  [WARN] trainval directory not found: {trainval}")
        return {}

    session_frames: dict[str, list[Path]] = {}
    for session_dir in sorted(trainval.iterdir()):
        if not session_dir.is_dir():
            continue
        cam_dir = session_dir / "CAM_F0"
        if not cam_dir.is_dir():
            continue
        frames = sorted(cam_dir.glob("*.jpg"))
        if len(frames) >= min_frames:
            session_frames[session_dir.name] = frames

    return session_frames


def generate_triplets(
    session_frames: dict[str, list[Path]],
    max_samples_per_session: int = 10,
) -> list[dict]:
    """对每个 session 做滑动窗口，生成 (current, future_0.5s, future_1.0s) 三元组。

    每个 session 最多取 max_samples_per_session 组（即 max_samples_per_session + 2 帧）。
    步长 1，每 3 帧一组。
    """
    records: list[dict] = []

    for session_id, frames in sorted(session_frames.items()):
        needed_frames = max_samples_per_session + 2
        session_frames_slice = frames[:needed_frames]

        for i in range(len(session_frames_slice) - 2):
            current = session_frames_slice[i]
            f0 = session_frames_slice[i + 1]
            f1 = session_frames_slice[i + 2]

            records.append({
                "current": str(current),
                "future_0.5s": str(f0),
                "future_1.0s": str(f1),
            })

    return records


def make_images_rel_path(frame_path: str, navsim_root: Path) -> str:
    """将绝对路径转为 images/ 相对路径。

    输入: /root/.../navsim_v1.1_all/dataset/sensor_blobs/trainval/{scene}/CAM_F0/{frame}.jpg
    输出: images/{scene}/CAM_F0/{frame}.jpg
    """
    fp = Path(frame_path)
    try:
        rel = fp.relative_to(navsim_root / "dataset" / "sensor_blobs" / "trainval")
        return str(Path("images") / rel)
    except ValueError:
        # 不在标准路径下，直接用 basename 组装
        parts = fp.parts
        # 找到 trainval 后面的部分
        try:
            idx = parts.index("trainval")
            scene_part = os.path.join(*parts[idx + 1:])
            return str(Path("images") / scene_part)
        except (ValueError, IndexError):
            return str(Path("images") / fp.name)


def create_symlinks(
    records: list[dict],
    navsim_root: Path,
    output_dir: Path,
) -> int:
    """为三元组中的帧创建 images/ 软链接。

    软链接指向 navsim_root 中的原始文件。
    输出结构: output_dir/images/{scene}/CAM_F0/{frame}.jpg
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    seen_links: set[str] = set()
    symlink_count = 0

    for rec in records:
        for key in ("current", "future_0.5s", "future_1.0s"):
            src_path = Path(rec[key])
            if not src_path.is_absolute():
                src_path = navsim_root / src_path

            if not src_path.exists():
                continue

            # 计算相对路径
            rel = make_images_rel_path(str(src_path), navsim_root)
            link_path = output_dir / rel

            if link_path in seen_links:
                continue
            seen_links.add(str(link_path))

            link_path.parent.mkdir(parents=True, exist_ok=True)
            if not link_path.exists():
                os.symlink(src_path, link_path)
                symlink_count += 1

    return symlink_count


# ═══════════════════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════════════════


def cmd_prepare(args: argparse.Namespace) -> None:
    """主命令：下载 + 扫描 + 生成 raw_index.jsonl + images/ 软链接。"""
    navsim_root = Path(args.navsim_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    random.seed(args.seed)

    print("=" * 60)
    print("NavSim Dataset Prepare")
    print("=" * 60)
    print(f"  navsim_root: {navsim_root}")
    print(f"  output_dir:  {output_dir}")
    print(f"  scan_only:   {args.scan_only}")
    if not args.scan_only:
        print(f"  num_sessions: {args.num_sessions}")
    print(f"  max_samples_per_session: {args.max_samples_per_session}")
    print()

    # ── Step 1: 扫描本地 session ──────────────────────────────────────────
    print("[1/5] Scanning local sessions ...")
    local_sessions = scan_local_sessions(navsim_root, min_frames=3)
    print(f"  Found {len(local_sessions)} local sessions with ≥3 frames")
    if local_sessions:
        total_frames = sum(len(v) for v in local_sessions.values())
        print(f"  Total frames: {total_frames}")

    if args.scan_only:
        # 只扫描，不下载
        if not local_sessions:
            print("  No local sessions with enough frames. Aborting.")
            sys.exit(1)
        selected_sessions = local_sessions
    else:
        # ── Step 2: 下载更多 session ──────────────────────────────────────
        print(f"\n[2/5] Ensuring {args.num_sessions} sessions ...")

        needed = args.num_sessions - len(local_sessions)
        if needed <= 0:
            print(f"  Already have {len(local_sessions)} sessions, no download needed")
            selected_sessions = local_sessions
        else:
            print(f"  Need to download {needed} more sessions")

            # 加载 shard index
            index = load_shard_index()
            session_to_shards = build_shard_map(index)

            # 收集所有可用 session（排除已本地的）
            all_sessions = sorted(set(session_to_shards.keys()))
            have_ids = set(local_sessions.keys())
            candidates = [s for s in all_sessions if s not in have_ids]

            # 从候选 session 中挑选：按 shard 分组，优先选能在一个 shard 内拿到更多 session 的
            # 策略：按 shard 中候选 session 数量降序，逐个 shard 拿 session，直到凑够 needed
            def frame_count_from_name(session_name: str) -> int:
                """从 session 名提取帧数：xxx_00005_00204 → 200"""
                parts = session_name.split("_")
                if len(parts) >= 3:
                    try:
                        start = int(parts[-2])
                        end = int(parts[-1])
                        return end - start + 1
                    except ValueError:
                        pass
                return 0

            # 按 shard 分组：shard_name → [候选 session 列表]
            # 同一 session 可能跨多个 shard，取第一个命中的
            candidates_set = set(candidates)
            shard_candidates: dict[str, list[str]] = defaultdict(list)
            for sess in candidates:
                for shard in session_to_shards.get(sess, []):
                    shard_candidates[shard].append(sess)
                    break  # 一个 session 只归到第一个 shard

            # 按 shard 内 session 数量降序，优先选 session 多的 shard
            shard_order = sorted(shard_candidates.keys(), key=lambda s: len(shard_candidates[s]), reverse=True)

            to_download: list[str] = []
            seen: set[str] = set()
            for shard in shard_order:
                for sess in shard_candidates[shard]:
                    if sess not in seen:
                        # 在同一个 shard 内，按帧数降序排列
                        to_download.append(sess)
                        seen.add(sess)
                    if len(to_download) >= needed:
                        break
                if len(to_download) >= needed:
                    break

            print(f"  Selected {len(to_download)} sessions to download")
            print(f"  Frame range: {frame_count_from_name(to_download[0])}-{frame_count_from_name(to_download[-1])}")

            # 按 shard 分组下载
            shard_sessions: dict[str, list[str]] = defaultdict(list)
            for sess in to_download:
                for shard in session_to_shards.get(sess, []):
                    shard_sessions[shard].append(sess)
                    break  # 一个 session 只归到第一个 shard

            print(f"\n  Will download from {len(shard_sessions)} shard(s)")
            for shard, sessions in sorted(shard_sessions.items()):
                print(f"    {shard}: {len(sessions)} sessions")

            # 下载（按 shard 分组，每个 shard 只下载一次）
            for shard_name, sessions in sorted(shard_sessions.items()):
                print(f"\n  --- Processing {shard_name} ---")
                results = download_and_extract_shard(
                    shard_name, sessions, navsim_root,
                    max_frames=args.max_samples_per_session + 2,
                )
                total = sum(results.values())
                print(f"  → {shard_name}: {total} files extracted for {len(sessions)} sessions")

            # 重新扫描本地
            print(f"\n  Re-scanning local sessions after download...")
            selected_sessions = scan_local_sessions(navsim_root, min_frames=3)
            print(f"  Now have {len(selected_sessions)} sessions with ≥3 frames")

    # ── Step 3: 生成三元组 ────────────────────────────────────────────────
    print(f"\n[3/5] Generating triplets (max {args.max_samples_per_session} triplets/session) ...")
    all_records = generate_triplets(selected_sessions, max_samples_per_session=args.max_samples_per_session)
    print(f"  Generated {len(all_records)} triplets from {len(selected_sessions)} sessions")

    if not all_records:
        print("  ERROR: No triplets generated. Check your data.")
        sys.exit(1)

    sampled = all_records

    # ── Step 4: 输出 ──────────────────────────────────────────────────────
    print(f"\n[4/5] Writing output ...")

    # 5a. 创建软链接
    print(f"  Creating symlinks in {output_dir / 'images'}/ ...")
    symlink_count = create_symlinks(sampled, navsim_root, output_dir)
    print(f"    Created {symlink_count} symlinks")

    # 5b. 将三元组中的路径转为 images/ 相对路径
    output_records = []
    for rec in sampled:
        output_records.append({
            "current": make_images_rel_path(rec["current"], navsim_root),
            "future_0.5s": make_images_rel_path(rec["future_0.5s"], navsim_root),
            "future_1.0s": make_images_rel_path(rec["future_1.0s"], navsim_root),
        })

    # 5c. 写入 raw_index.jsonl
    out_path = output_dir / "raw_index.jsonl"
    with open(out_path, "w") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(output_records)} records to {out_path}")

    if output_records:
        print(f"  Example: {json.dumps(output_records[0], indent=2)}")

    # ── 统计 ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Done! Summary:")
    print(f"  Sessions used: {len(selected_sessions)}")
    print(f"  Triplets generated: {len(output_records)}")
    print(f"  Symlinks: {symlink_count}")
    print(f"  Output: {out_path}")
    print(f"  Next: python -m scripts.prepare_stage05_data encode \\")
    print(f"          --input {out_path} \\")
    print(f"          --emu35_path /path/to/Emu3.5-VisionTokenizer \\")
    print(f"          --output {output_dir / 'data.jsonl'} \\")
    print(f"          --data_root {output_dir}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NavSim 数据集下载 + prepare — 下载 session → raw_index.jsonl + images/ 软链接",
    )
    parser.add_argument("--navsim_root", type=str, required=True,
                        help="NavSim 数据集根目录（包含 dataset/sensor_blobs/trainval/）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出数据集目录（将创建 images/ 和 raw_index.jsonl）")
    parser.add_argument("--max_samples_per_session", type=int, default=10,
                        help="每个 session 最多取多少组三元组（默认：10，即每个 session 取 12 帧）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认：42）")

    # 下载控制
    parser.add_argument("--num_sessions", type=int, default=20,
                        help="确保的 session 数量（默认：20，不足时下载）")
    parser.add_argument("--scan_only", action="store_true",
                        help="只扫描本地已有数据，不下载")

    args = parser.parse_args()
    cmd_prepare(args)


if __name__ == "__main__":
    main()