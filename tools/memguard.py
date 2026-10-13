#!/usr/bin/env python3
"""memguard: run a command under a project-wide RAM cap, with slow restarts.

Usage:
    .venv/bin/python tools/memguard.py --limit-gb 200 --parallel 8 --tag exp1 -- <cmd> [args...]

What it does:
  * Launches <cmd> in its own session (process group), exporting MEMGUARD_PARALLEL so the
    child can size its worker pool, and MEMGUARD_ATTEMPT (0, 1, 2, ...).
  * Every --interval seconds it measures PROJECT memory = RSS of the child process tree
    + RSS of every other process belonging to this project (cmdline/cwd under the project
    root, e.g. other experiments or our private Ollama server).
  * If PROJECT memory >= --limit-gb: kills the child group (SIGTERM, then SIGKILL), logs the
    event, waits a cooldown that grows with each restart, halves the parallelism and
    restarts. Child commands must be resumable (on-disk caches/checkpoints).
  * It also kills/restarts if the machine as a whole is about to swap
    (available memory < --min-free-gb), since the host is shared.

Events are appended to <project>/logs/memguard.log (JSON lines).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import psutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
GB = 1024 ** 3


def log(event: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    event = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    with open(os.path.join(LOG_DIR, "memguard.log"), "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[memguard] {event}", file=sys.stderr, flush=True)


def _belongs_to_project(p: psutil.Process) -> bool:
    try:
        cmd = " ".join(p.cmdline())
        if PROJECT_ROOT in cmd:
            return True
        cwd = p.cwd()
        return cwd.startswith(PROJECT_ROOT)
    except (psutil.Error, OSError):
        return False


def project_rss_bytes(extra_roots=()) -> int:
    """RSS of all processes of this project (deduplicated by pid). One batched pass over the
    process table (cmdline/cwd/rss/ppid fetched together) keeps the monitor itself cheap."""
    me = os.getpid()
    info = {}
    try:
        it = list(psutil.process_iter(["pid", "ppid", "cmdline", "cwd", "memory_info"]))
    except Exception:  # noqa: BLE001 - macOS sysctl permission races must never kill the guard
        it = []
    for p in it:
        try:
            d = p.info
            info[d["pid"]] = d
        except Exception:  # noqa: BLE001
            continue
    members = set(r for r in extra_roots if r in info)
    for pid, d in info.items():
        if pid == me:
            continue
        cmd = " ".join(d.get("cmdline") or [])
        cwd = d.get("cwd") or ""
        if PROJECT_ROOT in cmd or cwd.startswith(PROJECT_ROOT):
            members.add(pid)
    # include descendants of members
    children = {}
    for pid, d in info.items():
        children.setdefault(d.get("ppid"), []).append(pid)
    stack = list(members)
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in members:
                members.add(c)
                stack.append(c)
    total = 0
    for pid in members:
        mi = info.get(pid, {}).get("memory_info")
        if mi is not None:
            total += mi.rss
    return total


def kill_group(proc: subprocess.Popen, grace: float = 20.0):
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < grace:
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-gb", type=float, default=200.0)
    ap.add_argument("--min-free-gb", type=float, default=24.0,
                    help="also restart if system available memory drops below this")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--min-parallel", type=int, default=1)
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--cooldown", type=float, default=90.0, help="base cooldown seconds")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--tag", default="job")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("missing command after --")

    parallel = a.parallel
    attempt = 0
    while True:
        env = dict(os.environ, MEMGUARD_PARALLEL=str(parallel), MEMGUARD_ATTEMPT=str(attempt),
                   MEMGUARD_LIMIT_GB=str(a.limit_gb))
        log({"tag": a.tag, "event": "start", "attempt": attempt, "parallel": parallel, "cmd": cmd})
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        peak = 0
        breach = None
        while proc.poll() is None:
            time.sleep(a.interval)
            rss = project_rss_bytes(extra_roots=(proc.pid,))
            peak = max(peak, rss)
            avail = psutil.virtual_memory().available
            if rss >= a.limit_gb * GB:
                breach = f"project RSS {rss / GB:.1f} GB >= limit {a.limit_gb} GB"
            elif avail < a.min_free_gb * GB:
                breach = f"system available {avail / GB:.1f} GB < {a.min_free_gb} GB"
            if breach:
                log({"tag": a.tag, "event": "breach", "reason": breach, "attempt": attempt})
                kill_group(proc)
                break
        rc = proc.returncode
        log({"tag": a.tag, "event": "exit", "rc": rc, "attempt": attempt,
             "peak_project_gb": round(peak / GB, 2), "breach": breach})
        if breach is None:
            sys.exit(rc)
        attempt += 1
        if attempt > a.max_restarts:
            log({"tag": a.tag, "event": "give_up", "attempt": attempt})
            sys.exit(3)
        parallel = max(a.min_parallel, parallel // 2)
        cool = a.cooldown * attempt
        log({"tag": a.tag, "event": "cooldown", "seconds": cool, "next_parallel": parallel})
        time.sleep(cool)


if __name__ == "__main__":
    main()
