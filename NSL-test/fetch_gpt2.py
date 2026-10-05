"""Resumable multi-connection downloader for the OpenAI GPT-2 TF checkpoints.

Usage:
    python fetch_gpt2.py --size 124M --dir models/124M --parts 8

Safe to interrupt (Ctrl+C) and re-run: it resumes from whatever is already on disk
(each part file's current size is used as the resume offset).
"""
import argparse
import math
import os
import sys
import threading
import time

import requests

BASE = "https://openaipublic.blob.core.windows.net/gpt-2/models"
SMALL_FILES = [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
]
BIG_FILE = "model.ckpt.data-00000-of-00001"
LOCK = threading.Lock()


def log(msg):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def head_size(session, url, tries=15):
    last = None
    for _ in range(tries):
        try:
            r = session.head(url, timeout=(30, 60), allow_redirects=True)
            r.raise_for_status()
            return int(r.headers["content-length"])
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3)
    raise RuntimeError(f"HEAD failed for {url}: {type(last).__name__}: {last}")


def fetch_sequential(session, url, path, expected):
    """Download a small file with resume; returns True when complete."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have == expected:
        log(f"skip (complete) {os.path.basename(path)} {expected} bytes")
        return True
    while have < expected:
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 60)) as r:
                if have and r.status_code != 206:
                    raise RuntimeError(f"server ignored Range (status {r.status_code})")
                r.raise_for_status()
                with open(path, "ab") as f:
                    for chunk in r.iter_content(1 << 16):
                        if not chunk:
                            continue
                        f.write(chunk)
                        have += len(chunk)
                        if have >= expected:
                            break
        except Exception as e:  # noqa: BLE001
            log(f"retry {os.path.basename(path)} at {have}/{expected}: {type(e).__name__}: {str(e)[:100]}")
            time.sleep(3)
    log(f"done {os.path.basename(path)} {have} bytes")
    return True


def part_worker(session, url, part_path, start, end, counters, stop):
    total = end - start + 1
    while not stop.is_set():
        done = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if done >= total:
            with LOCK:
                counters[part_path] = total
            return
        headers = {"Range": f"bytes={start + done}-{end}"}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 90)) as r:
                if r.status_code == 206:
                    cr = r.headers.get("content-range", "")
                    if cr and not cr.startswith(f"bytes {start + done}-"):
                        raise RuntimeError(f"bad content-range {cr}")
                elif r.status_code == 200 and start + done != 0:
                    raise RuntimeError("server ignored Range")
                r.raise_for_status()
                with open(part_path, "ab") as f:
                    for chunk in r.iter_content(1 << 16):
                        if stop.is_set():
                            return
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        with LOCK:
                            counters[part_path] = done
                        if done >= total:
                            break
        except Exception as e:  # noqa: BLE001
            log(f"part {os.path.basename(part_path)} stalled at {done}/{total}: "
                f"{type(e).__name__}: {str(e)[:100]} -> resuming")
            time.sleep(3)


def monitor(counters, total, stop, t0):
    while not stop.wait(20):
        got = sum(counters.values())
        elapsed = time.time() - t0
        speed = got / elapsed if elapsed else 0
        eta = (total - got) / speed if speed > 0 else float("inf")
        log(f"progress {got}/{total} = {100.0 * got / total:.1f}%  "
            f"avg {speed / 1024:.1f} KB/s  eta {eta / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="124M")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--parts", type=int, default=8)
    args = ap.parse_args()

    model_dir = args.dir or os.path.join("models", args.size)
    os.makedirs(model_dir, exist_ok=True)
    url_base = f"{BASE}/{args.size}"
    session = requests.Session()
    session.headers["User-Agent"] = "curl/8.0"

    log(f"target dir: {os.path.abspath(model_dir)}")

    for name in SMALL_FILES:
        url = f"{url_base}/{name}"
        expected = head_size(session, url)
        fetch_sequential(session, url, os.path.join(model_dir, name), expected)

    url = f"{url_base}/{BIG_FILE}"
    final = os.path.join(model_dir, BIG_FILE)
    total = head_size(session, url)
    log(f"{BIG_FILE} total size = {total} bytes")

    if os.path.exists(final) and os.path.getsize(final) == total:
        log("big file already complete, nothing to do")
        return 0

    n_parts = max(1, args.parts)
    chunk = math.ceil(total / n_parts)
    parts = []
    for i in range(n_parts):
        start = i * chunk
        end = min(total - 1, start + chunk - 1)
        if start > end:
            break
        parts.append((os.path.join(model_dir, f".{BIG_FILE}.part{i}"), start, end))

    counters = {p: (os.path.getsize(p) if os.path.exists(p) else 0) for p, _, _ in parts}
    stop = threading.Event()
    t0 = time.time()
    threads = [threading.Thread(target=part_worker, args=(session, url, p, s, e, counters, stop), daemon=True)
               for p, s, e in parts]
    for t in threads:
        t.start()
    mon = threading.Thread(target=monitor, args=(counters, total, stop, t0), daemon=True)
    mon.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop.set()
        log("interrupted - partial data kept, re-run to resume")
        return 130
    stop.set()

    got = sum(counters.values())
    if got != total:
        log(f"incomplete: {got}/{total} - re-run to resume")
        return 1

    log("assembling parts ...")
    with open(final, "wb") as out:
        for p, _, _ in parts:
            with open(p, "rb") as src:
                while True:
                    buf = src.read(1 << 22)
                    if not buf:
                        break
                    out.write(buf)
            os.remove(p)
    final_size = os.path.getsize(final)
    log(f"final file size = {final_size} (expected {total})")
    if final_size != total:
        log("SIZE MISMATCH - delete the file and re-run")
        return 1
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
