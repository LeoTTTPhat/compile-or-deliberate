# synthesised by GP (B5); episode evaluations 4996; generation 1; seed 3
# training-batch success 0.843 (A: 12/12, P: 16/16, Q: 12/12, S: 18/18, SN: 12/25)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if SL_avg < 49.449 -> rate_cut_25
    if not mem.get("R1") and vetoed != "R1" and f["SL_avg"] < 49.449:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "unknown", "why": "SL_avg < 49.449", "action": {"type": "procedure", "name": "rate_cut_25"}}
    # R2: if not armed_overrides -> min_stable_park
    if not mem.get("R2") and vetoed != "R2" and not f["armed_overrides"]:
        mem["R2"] = 1
        mem["last"] = "R2"
        return {"rule": "R2", "diagnosis": "unknown", "why": "not armed_overrides", "action": {"type": "procedure", "name": "min_stable_park"}}
    # R3: if XLEVEL_HI -> subst_P
    if not mem.get("R3") and vetoed != "R3" and "reactor_pressure" not in f["substituted"] and f["A_XLEVEL_HI"]:
        mem["R3"] = 1
        mem["last"] = "R3"
        return {"rule": "R3", "diagnosis": "sensor_fault", "why": "XLEVEL_HI", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    # R4: if T_slope > -0.017028 -> arm_both
    if not mem.get("R4") and vetoed != "R4" and not (f["armed_composition"] and f["armed_overrides"]) and f["T_slope"] > -0.017028:
        mem["R4"] = 1
        mem["last"] = "R4"
        return {"rule": "R4", "diagnosis": "unknown", "why": "T_slope > -0.017028", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    # R5: if not TEMP_HIHI and T_avg > 120.36 -> reactor_pressure-50
    if not mem.get("R5") and vetoed != "R5" and f["P_sp"] > 2500.0 and not f["A_TEMP_HIHI"] and f["T_avg"] > 120.36:
        mem["R5"] = 1
        mem["last"] = "R5"
        return {"rule": "R5", "diagnosis": "unknown", "why": "not TEMP_HIHI and T_avg > 120.36", "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": max(2500.0, f["P_sp"] - 50.0)}}}
    # R6: if XL_slope > 1.3255 and n_alarms < 3.5 -> arm_composition
    if not mem.get("R6") and vetoed != "R6" and not f["armed_composition"] and f["XL_slope"] > 1.3255 and f["n_alarms"] < 3.5:
        mem["R6"] = 1
        mem["last"] = "R6"
        return {"rule": "R6", "diagnosis": "unknown", "why": "XL_slope > 1.3255 and n_alarms < 3.5", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    return None
