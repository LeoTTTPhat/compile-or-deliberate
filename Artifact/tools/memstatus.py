#!/usr/bin/env python3
"""Print this project's current memory use (RSS) and the host's available memory."""
import os
import sys

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memguard import GB, PROJECT_ROOT, _belongs_to_project, project_rss_bytes  # noqa: E402

if __name__ == "__main__":
    rows = []
    for p in psutil.process_iter(["pid", "name"]):
        if p.pid != os.getpid() and _belongs_to_project(p):
            try:
                rows.append((p.memory_info().rss, p.pid, " ".join(p.cmdline())[:120]))
            except psutil.Error:
                pass
    for rss, pid, cmd in sorted(rows, reverse=True)[:15]:
        print(f"{rss / GB:7.2f} GB  {pid:>7}  {cmd}")
    vm = psutil.virtual_memory()
    print(f"PROJECT total RSS: {project_rss_bytes() / GB:.2f} GB (cap 200 GB) | "
          f"host available: {vm.available / GB:.1f} GB of {vm.total / GB:.0f} GB | root={PROJECT_ROOT}")
