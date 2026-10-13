"""Cached, thread-safe client for the project's private Ollama server.

Every call is keyed by (model, messages, options, format) and stored in SQLite, so a killed
or restarted experiment never pays twice for the same LLM call. With temperature > 0, pass
distinct `seed`s to get distinct (but still cached and reproducible) samples.

    from tools.llm import LLM
    llm = LLM("qwen2.5:7b")
    r = llm.chat([{"role": "user", "content": "hi"}], max_tokens=64)
    r["text"], r["completion_tokens"], r["cached"]
"""
import hashlib
import json
import os
import sqlite3
import threading
import time

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HOST = os.environ.get("LLM_HOST", "http://127.0.0.1:11435")
DEFAULT_CACHE = os.environ.get("LLM_CACHE", os.path.join(PROJECT_ROOT, "cache", "llm_cache.sqlite"))

_db_lock = threading.Lock()
_conns = {}


def _conn(path):
    key = (path, threading.get_ident())
    c = _conns.get(key)
    if c is None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        c = sqlite3.connect(path, timeout=60, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, model TEXT, resp TEXT, "
                  "ptok INT, ctok INT, latency REAL, ts REAL, tag TEXT)")
        c.commit()
        _conns[key] = c
    return c


class LLM:
    def __init__(self, model, host=DEFAULT_HOST, cache_path=DEFAULT_CACHE, num_ctx=16384,
                 timeout=900, retries=4):
        self.model, self.host, self.cache_path = model, host.rstrip("/"), cache_path
        self.num_ctx, self.timeout, self.retries = num_ctx, timeout, retries

    def _key(self, payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def chat(self, messages, temperature=0.0, seed=0, max_tokens=512, fmt=None, tag=None,
             top_p=None, stop=None, use_cache=True):
        options = {"temperature": temperature, "seed": seed, "num_predict": max_tokens,
                   "num_ctx": self.num_ctx}
        if top_p is not None:
            options["top_p"] = top_p
        if stop:
            options["stop"] = stop
        payload = {"model": self.model, "messages": messages, "options": options, "stream": False,
                   "keep_alive": "30m"}
        if fmt is not None:
            payload["format"] = fmt
        k = self._key(payload)
        c = _conn(self.cache_path)
        if use_cache:
            row = c.execute("SELECT resp, ptok, ctok, latency FROM cache WHERE k=?", (k,)).fetchone()
            if row:
                return {"text": row[0], "prompt_tokens": row[1], "completion_tokens": row[2],
                        "latency": row[3], "cached": True}
        err = None
        for attempt in range(self.retries):
            try:
                t0 = time.time()
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
                r.raise_for_status()
                d = r.json()
                lat = time.time() - t0
                text = d["message"]["content"]
                ptok, ctok = d.get("prompt_eval_count", 0), d.get("eval_count", 0)
                with _db_lock:
                    c.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?,?)",
                              (k, self.model, text, ptok, ctok, lat, time.time(), tag))
                    c.commit()
                return {"text": text, "prompt_tokens": ptok, "completion_tokens": ctok,
                        "latency": lat, "cached": False}
            except Exception as e:  # network hiccup / server restart
                err = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.retries} retries: {err}")

    def json_chat(self, messages, schema=None, **kw):
        """Chat with JSON-constrained output; returns (parsed_or_None, raw_response)."""
        r = self.chat(messages, fmt=schema if schema is not None else "json", **kw)
        try:
            return json.loads(r["text"]), r
        except json.JSONDecodeError:
            return None, r


def usage_summary(cache_path=DEFAULT_CACHE):
    c = _conn(cache_path)
    return c.execute("SELECT model, tag, COUNT(*), SUM(ptok), SUM(ctok), AVG(latency) FROM cache "
                     "GROUP BY model, tag").fetchall()
