"""Prompt material shared by the offline playbook synthesiser and the online deliberator.

Part A: process-agnostic operating doctrine (objective ordering, restraint, act early on
evidence, verify consistency before trusting an instrument).
Part B: process description generated from the plant configuration: equipment, regulatory
layer, alarm table, action grammar with bounds, pre-engineered components, soft sensors and a
HAZOP-style catalogue of ANTICIPATED deviations (no parameters, no scenario labels).
Both arms receive identical Part A + Part B, so they differ only in when LLM compute is spent.
"""
from __future__ import annotations

import hashlib

from .actions import BOUNDS, DIAGNOSES, PROCEDURES, SUBSTITUTABLE
from .alarms import ALARMS

PART_A = """You are the abnormal-situation-management (ASM) layer of a continuous chemical plant. A decentralized
regulatory control layer (PI loops) runs underneath you and keeps the plant at its operating point; you
act only through a bounded supervisory interface (setpoints, arming pre-engineered control components,
pre-audited procedures, taking a lying transmitter out of service, no action, orderly shutdown). An
independent safety-instrumented system trips the plant at hard limits; a trip is the worst outcome.
Objectives in strict priority: (1) process safety: never reach a trip limit; (2) equipment integrity;
(3) plant availability: an orderly shutdown also counts as failure, use it only if nothing else can work;
(4) product quality: deliver the commanded product grade; (5) economics: keep production near target.
Doctrine:
- Diagnose from accumulation balances, trends and cross-instrument consistency, not from a single reading.
- Act EARLY on credible evidence: many upsets become unrecoverable 1-2 h after onset even though alarms
  annunciate late. A rising trend with a short time-to-limit is evidence.
- Withhold action when the regulatory layer is coping (margins large and stable, no worsening trend):
  unnecessary interventions can destabilise vessel inventories.
- A transmitter that disagrees with its physically coupled neighbours may be lying; a loop fed by a lying
  transmitter drives the true variable away while its reading looks normal. Prefer the pre-engineered soft
  sensor for that loop.
- Arming a control component is one-way and can induce inventory transients for 2-4 h; anticipate them.
- Prefer one decisive, well-chosen action over many small ones; respect cooldowns and the action budget."""


def _alarm_table() -> str:
    rows = []
    for a in ALARMS:
        if a.kind == "DEV":
            rows.append(f"{a.name}: |{a.measurement} - setpoint {a.setpoint}| > {a.limit:g} for {a.on_delay_h:g} h")
        else:
            rows.append(f"{a.name}: {a.measurement} {'>' if a.kind == 'HI' else '<'} {a.limit:g} (priority {a.priority})")
    return "; ".join(rows)


def _bounds() -> str:
    return "; ".join(f"{k} in [{b.lo:g}, {b.hi:g}], max move {b.max_delta:g}, cooldown {b.cooldown_h:g} h"
                     for k, b in BOUNDS.items())


PART_B = f"""PLANT: Tennessee Eastman process (revised model), mode 1. Gaseous reactants A, C, D, E (+ inert B) react in a
cooled two-phase reactor to liquid products G and H (byproduct F). Reactor effluent is cooled in a condenser,
flashed in a vapour-liquid separator; vapour is recycled by a compressor (with a purge for inerts), liquid
goes to a stripper whose underflow is the product. Nominal: reactor pressure 2705 kPa, reactor temperature
120.4 C, reactor level 75 %, separator level 50 %, stripper level 50 %, product %G 53.7 mol%, production
22.95 m3/h. HARD TRIP LIMITS (true values): reactor pressure 3000 kPa; reactor temperature 175 C; reactor,
separator and stripper levels outside their 0-100 % indicated ranges.
REGULATORY LAYER (Ricker decentralized PI): feed flows ratio-controlled to a production index; reactor pressure
-> purge; reactor level -> separator temperature setpoint (condenser cooling); reactor temperature -> reactor
cooling-water valve; separator level -> separator underflow; stripper level -> product valve. Two pre-engineered
components are AVAILABLE BUT DISARMED: composition_control (yA/yAC reactant-composition trims plus %G
feedforward/feedback; restores reactant balance and tracks product grade) and safety_overrides (high-pressure
override that sheds production as pressure approaches its limit; high-level override on the recycle valve).
ALARMS (on measured values): {_alarm_table()}.
SUPERVISORY ACTIONS (exactly one per decision, JSON):
- {{"type": "adjust_setpoints", "changes": {{field: absolute_target}}}} with {_bounds()}.
- {{"type": "reconfigure", "components": [...]}} components: composition_control, safety_overrides (one-way arm).
- {{"type": "procedure", "name": ...}} pre-audited: {", ".join(sorted(PROCEDURES))} (rate_cut_25/50 cut the production
  target by 25/50 %; depressurize_step lowers the pressure setpoint by 75 kPa; min_stable_park halves production
  then settles pressure at 2600 kPa).
- {{"type": "substitute_instrument", "measurement": m, "method": "soft"}} for m in {list(SUBSTITUTABLE)}: the loop
  uses the pre-engineered soft sensor instead of the transmitter (pressure: from separator/stripper pressure,
  compressor work, recycle flow; temperature: from cooling-water outlet temperature and filtered valve position).
- {{"type": "no_action"}};  {{"type": "shutdown"}} (orderly shutdown = failure; last resort).
Actions land 0.05 h after the decision; at most 12 actions per episode; every action passes a static
validator and a 6-h forward simulation on a nominal model (blind to faults) that vetoes actions predicted to
make things worse than holding.
ANTICIPATED ABNORMAL SITUATIONS (HAZOP catalogue; causes, not parameters): reactant-composition upsets in the
A/C feed (stream 4) and C-header pressure loss; loss or restriction of the A feed, especially at elevated
production; product-grade (%G) changes ordered by operations; reactor pressure or temperature transmitter
failures (bias, drift, frozen reading); changes of production target.
DIAGNOSIS VOCABULARY: {", ".join(DIAGNOSES)}."""

PROMPT_HASH = hashlib.sha256((PART_A + PART_B).encode()).hexdigest()[:16]

FEATURE_GLOSSARY = """FEATURES available in the dict `f` (all measured/derived signals; floats unless noted):
- f["t"] time (h); f["trigger"] str in {"new_alarm","alarm_hold","heartbeat","revision"}.
- For each key K in P (reactor pressure kPa), T (reactor temperature C), RL, SL, XL (reactor/separator/stripper
  level %), G (product %G mol%), PROD (production m3/h), SP (separator pressure), XP (stripper pressure),
  ST (separator temp), XT (stripper temp), PURGE, FA, FD, FE, FAC (feed flows A, D, E, A+C), RECY (recycle),
  RFEED (reactor feed), CW_R / CW_C (reactor / condenser cooling-water outlet temperature), WORK (compressor work),
  XA / XC (reactor-feed A / C mol%):
    f[K] latest value, f[K+"_avg"] 15-min mean, f[K+"_slope"] slope per hour over the last hour, f[K+"_slope_se"]
    its standard error, f[K+"_dev_nom"] deviation of the 15-min mean from the nominal operating point.
  For regulated K in P, T, RL, SL, XL, G, PROD also f[K+"_sp"] current setpoint and f[K+"_err"] mean minus setpoint.
- For K in P, T, RL, SL, XL: f[K+"_margin"] margin to the trip limit (same units) and f[K+"_ttl"] projected
  time-to-limit in hours (float('inf') if not approaching).
- f["yA"], f["yAC"] reactor-feed A and A+C composition; f["yA_sp"], f["yAC_sp"]; f["yA_dev_nom"].
- Valves (0-100 %): f["mv_d_feed"], f["mv_e_feed"], f["mv_a_feed"], f["mv_ac_feed"], f["mv_compressor_recycle"],
  f["mv_purge"], f["mv_s_underflow"], f["mv_x_underflow"], f["mv_x_steam"], f["mv_r_cooling_water"],
  f["mv_s_cooling_water"], f["mv_r_agitator_speed"]; each also with suffix "_std" (1-h standard deviation; a
  value near 0 means the valve is not moving). f["saturated"] list of saturated valve names; f["n_saturated"].
- Consistency residuals: f["dP_rs"] reactor-minus-separator pressure difference minus its nominal (kPa);
  f["P_resid"] reactor pressure reading minus its soft-sensor estimate (kPa); f["T_resid"] reactor temperature
  reading minus its soft-sensor estimate (C); f["P_soft"], f["T_soft"] the estimates.
- Alarms: f["alarms"] sorted list of active alarm names; f["alarm_ages"] dict name -> hours active;
  f["A_<NAME>"] bool per alarm; f["n_alarms"].
- State: f["armed_composition"], f["armed_overrides"] bools; f["substituted"] list of instruments out of service;
  f["cmd_pct_g"], f["cmd_production"] operator targets; f["G_cmd_err"], f["PROD_cmd_err"] delivered minus target.
- During a revision call: f["rejected"] the rejected action dict and f["reasons"] list of rejection reasons."""
