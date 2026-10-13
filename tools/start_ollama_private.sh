#!/bin/zsh
# Start (or reuse) this project's private Ollama server on 127.0.0.1:11435.
# It shares the model store with the user's main server (port 11434), but it batches
# requests (NUM_PARALLEL=8), which gives about 4x the aggregate throughput on the
# contended GPU.
# CONTEXT_LENGTH is capped at 8192: the server default (262144) would allocate a huge KV
# cache and use a lot of RAM.
# Its cwd is the project root, so memguard/memstatus count its RAM toward the 200 GB cap.
ROOT=${0:A:h:h}
cd "$ROOT" || exit 1
mkdir -p logs
if curl -s --max-time 2 127.0.0.1:11435/api/version >/dev/null; then
  echo "private ollama already running"; exit 0
fi
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-8} OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS:-4} \
OLLAMA_KEEP_ALIVE=30m OLLAMA_FLASH_ATTENTION=1 OLLAMA_CONTEXT_LENGTH=16384 \
nohup ~/.local/bin/ollama serve > logs/ollama_private.log 2>&1 &
echo $! > logs/ollama_private.pid
sleep 3
curl -s 127.0.0.1:11435/api/version && echo " started pid $(cat logs/ollama_private.pid)"
