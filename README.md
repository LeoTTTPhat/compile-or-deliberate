# Reproduction artifact: compile-or-deliberate agents for abnormal situation management

This artifact contains everything needed to reproduce the experiments and analyses of a study of small,
on-premises language-model agents that manage abnormal situations on the revised Tennessee Eastman process:
the simulation harness and all supervisors, the scenario suites, every recorded episode, the synthesised
playbooks, the language-model call cache and the analysis scripts. All language models run locally through
Ollama; no cloud API is used. Run every command from the folder that contains this file.

## Contents

```
README.md                          this file
MANIFEST.txt                       sha256, size and path of every file, and what was excluded and why
tools/                             cached language-model client (llm.py), memory guard (memguard.py, memstatus.py),
                                   launcher of the private Ollama server (start_ollama_private.sh)
cache/llm_cache.sqlite             every language-model call: request key, response, token counts, latency
c10_compile_or_deliberate/
  code/                            run_all.sh (end-to-end reproduction), requirements.txt, LICENSE
    cod/                           the harness and all supervisors (importable package)
    experiments/                   experiment drivers and analysis scripts
    scripts/                       soft sensors, suites, table and figure generators, release exporter
    tests/                         pytest suite (CPU only)
  data/                            soft sensors, nominal operating point, FDD training data and fitted models,
                                   suites/ (scenario suites with content hashes)
  results/
    screen/, screen_101/           headroom screens: labelled scenarios with the oracle's plans
    synth/, synth_archive/         synthesis runs: playbook checkpoints, metadata, training batches, run logs
    arms/                          one JSON per episode (decisions, outcome, tokens, latency)
    coverage/, stream/, preview/   failure-rate experiments, compile-back stream, preview checkpoints
  results_summary/                 episodes.csv (one row per episode), analysis.json, repro_env.json
                                   (Ollama version, model digests, package versions)
  prompts/                         rendered prompt strings and PROMPT_HASH
```

Episode files keep the per-step plant trace only for the case-study scenario (SN0003); no analysis needs the
traces of the other episodes, and dropping them reduces the artifact from about 3 GB to about 0.4 GB.

## Requirements

* Python 3.12 with the packages pinned in `c10_compile_or_deliberate/code/requirements.txt`. The simulator
  (`tep-studio==0.2.0`) is a pip dependency.
* No GPU is needed to reproduce the analyses or to replay the language-model experiments: every call is in
  `cache/llm_cache.sqlite`. Making new calls needs Ollama 0.33.3 with `qwen3-coder:30b`, `qwen2.5-coder:7b`,
  `qwen2.5:7b` and `qwen2.5:14b-instruct-q4_K_M` (Q4_K_M quantisation); the exact model digests are in
  `results_summary/repro_env.json`.
* The CPU stages took hours on a 32-core workstation; plan for about 0.5 GB of free disk space.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r c10_compile_or_deliberate/code/requirements.txt
export PYTHONPATH=c10_compile_or_deliberate/code
PY=.venv/bin/python
```

## 1. Check the installation (a few minutes)

```bash
$PY -m pytest -q c10_compile_or_deliberate/code/tests
```

## 2. Regenerate all tables, statistics and figures from the recorded episodes (minutes, CPU only)

```bash
export COD_GEN=outputs/generated COD_FIG=outputs/figures
$PY c10_compile_or_deliberate/code/experiments/analyze.py --workers 4 --case SN0003 --no-pending
$PY c10_compile_or_deliberate/code/experiments/final_stats.py
$PY c10_compile_or_deliberate/code/experiments/figures_revision.py
$PY c10_compile_or_deliberate/code/scripts/make_playbook_figure.py
$PY c10_compile_or_deliberate/code/scripts/gen_appendix_tables.py
$PY c10_compile_or_deliberate/code/scripts/gen_repro.py --offline
```

`outputs/generated/` receives the result tables, the paired statistics and the files that hold every reported
number as a named macro (written by `analyze.py` and `final_stats.py`);
`outputs/figures/` receives the figures as vector PDFs. `analysis.json` summarises every arm in machine-readable
form. `gen_repro.py --offline` writes the reproducibility record (models, digests, software, seeds, method
parameters) from `results_summary/repro_env.json` without contacting Ollama.

## 3. Re-run the experiments

`c10_compile_or_deliberate/code/run_all.sh` runs every stage in order; `STAGES="..."` selects stages:

```bash
bash c10_compile_or_deliberate/code/run_all.sh                       # all stages
STAGES="playbooks revision analysis" bash c10_compile_or_deliberate/code/run_all.sh
```

Stages: `setup softsensors screens suites fdd baselines gp synth online playbooks hybrid stream coverage revision
analysis`. Every step is resumable, because results are one JSON file per episode or checkpoint. A stage whose
files exist is skipped, so delete (or move) the corresponding `results/` sub-folder to recompute it. Every
language-model call goes through the cache, so the language-model stages replay exactly without a GPU; with an
empty cache they take days of decoding on one workstation-class GPU. Heavy steps run under `tools/memguard.py`.

Single experiments (run from this folder with `PY` and `PYTHONPATH` as above; `MG="$PY tools/memguard.py
--limit-gb 200"`; script paths below are relative to `c10_compile_or_deliberate/code/`):

| Result | Command |
|---|---|
| Soft sensors (`data/soft_sensors.json`) | `scripts/fit_soft_sensors.py` |
| Headroom screens | `$MG --parallel 16 --tag screen -- $PY experiments/screen.py 50` (seed 7, training) and `... screen.py 40 A,S,P,Q,SN,V,C,K 101` (seed 101, held-out source) |
| Held-out suites and hashes | `scripts/make_suites.py` |
| FDD classifier and novelty gate | `experiments/fdd_data.py synth`, `experiments/fdd_data.py heldout --n 15`, then `$PY -m cod.fdd train` |
| References and LLM-free baselines | `experiments/run_arms.py --arms weak,strong,B2a,B2b,fdd:default,search:nominal --suite alhazmi,frontier,test_id,test_ood,test_unrescuable --seeds 1,2,3` |
| Perfect-twin search | `run_arms.py --arms "search:perfect@gate=0" --policy '{"shadow_fidelity": "perfect"}' ...` |
| Genetic programming | `experiments/gp_run.py --seed <1,2,3> --evals 20000` |
| Language-model playbook synthesis | `scripts/launch_synth.sh qwen3-coder:30b <seed> main 400000`, or `experiments/synth_run.py --model qwen3-coder:30b --seed <s> --tag main --budget 400000 --checkpoints 25000,50000,100000,200000,400000` |
| Smaller coder | `synth_run.py --model qwen2.5-coder:7b --seed 1 --tag main ...` (stopped at about 188k tokens; checkpoints up to 100k are used) |
| Synthesis ablations | `--no-remedy --tag norem`; `--tune-budget 0 --tag notune`; `--one-shot --tag oneshot` |
| Online deliberation | `run_arms.py --arms onlineg:qwen3-coder:30b --suite test_id,test_ood,alhazmi,frontier --seeds 1,2,3`; token buckets with `onlineg:<model>@<tokens per episode>` |
| Playbook arms | `run_arms.py --arms playbook:<path to playbook_<ckpt>.py> --suite ...` |
| Hybrid agents | `run_arms.py --arms hybrid:<playbook.py>:<model>@<tokens>[:trig=abc,cool=1,maxesc=3]`; `experiments/grid.py --synth <run dir> --print` lists a grid |
| Compile-back stream | `experiments/stream_hybrid.py --playbook <final.py> --model qwen3-coder:30b --tokens 16000 --name cb_s1_final_16k` (control: `--no-compile`) |
| Failure-rate bounds | `experiments/coverage.py <playbook.py> --name <name>` |
| Calibration-only ablation | `experiments/calib_only.py --seed <s> --rounds 1` |
| Validation and confirmation suites | `experiments/make_confirm_suites.py` (`select_id`, `confirm_id`, `confirm_ood`), then the frozen arms on them with `run_arms.py --suite select_id,confirm_id,confirm_ood` |
| Robustness checks | compiled supervisors on the strong configuration: `run_arms.py --arms onstrong:<arm> --suite test_id,test_ood`; dense invocation: `--suite test_id_dense,test_ood_dense --policy '{"periodic_h": 0.5, "hold_interval_h": 0.5, "max_calls": 96}'` |

Arm grammar of `run_arms.py`: `weak | strong | hold | B2a | B2b | fdd:[model.pkl][@thr=..] |
search:{nominal,perfect}[@h=..,thr=..,gate=..] | online:<model>[@tokens] | onlineg:<model>[@tokens] |
playbook:<path.py>[@name] | hybrid:<path.py>:<model>@<tokens>[:opts] | onstrong:<arm>`. Paths inside arm specs
are relative to this folder.

## Notes

* **Memory guard.** `tools/memguard.py --limit-gb 200 --parallel N` caps the memory of a job and its children; on
  a breach it stops the job, waits, halves the parallelism and restarts it. All drivers are resumable.
* **Private Ollama server.** `tools/start_ollama_private.sh` starts `ollama serve` on 127.0.0.1:11435 with 8
  parallel sequences, flash attention and a 16,384-token context; every request uses `num_ctx=16384`. Override
  the server with `LLM_HOST` and the cache with `LLM_CACHE`.
* **Language-model cache.** `tools/llm.py` keys every request on (model, messages, options, format). With
  temperature above zero the sampling seed is part of the key. Re-querying a server is not guaranteed to
  reproduce completions bit for bit (batched GPU inference is not deterministic), so the cache is the record.
* **Multiprocessing.** macOS uses `spawn`; every script that creates a process pool has a `__main__` guard.
* **Latency.** The recorded wall-clock latencies were measured on a workstation shared with unrelated jobs and are
  therefore inflated; token counts are not affected.
* **Anonymisation.** Absolute paths were rewritten to paths relative to this folder, and identifying strings
  were removed. `MANIFEST.txt` lists every file with its sha256.

## Licence and third-party components

* The code is released under the MIT licence (`c10_compile_or_deliberate/code/LICENSE`).
* `tep-studio` (revised Tennessee Eastman kernel) is a pip dependency under BSD-3-Clause; the bundled process
  model keeps its original authors' terms. It is not vendored.
* The public harness of Alhazmi (2026) carries no licence and is not redistributed; the bounded interface,
  validator and anti-windup fix are independent implementations, and its 13 anchor and 5 frontier scenarios are
  re-encoded as scenario definitions in `cod/scenarios.py`.
* Model weights are not included; Ollama pulls them under their own licences.
