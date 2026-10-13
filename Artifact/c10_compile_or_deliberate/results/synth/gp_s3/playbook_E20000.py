# synthesised by GP (B5); episode evaluations 19998; generation 9; seed 3
# training-batch success 0.843 (A: 12/12, P: 16/16, Q: 12/12, S: 20/20, SN: 31/48)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if P_resid < -0.043017 and mv_r_cooling_water > 40.039 -> arm_both
    if not mem.get("R1") and vetoed not in ("R1", "R2", "R7") and not (f["armed_composition"] and f["armed_overrides"]) and f["P_resid"] < -0.043017 and f["mv_r_cooling_water"] > 40.039:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "unknown", "why": "P_resid < -0.043017 and mv_r_cooling_water > 40.039", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    # R2: if T_slope > -0.017028 -> arm_both
    if not mem.get("R2") and vetoed not in ("R1", "R2", "R7") and not (f["armed_composition"] and f["armed_overrides"]) and f["T_slope"] > -0.017028:
        mem["R2"] = 1
        mem["last"] = "R2"
        return {"rule": "R2", "diagnosis": "unknown", "why": "T_slope > -0.017028", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    # R3: if not RLEVEL_HI -> subst_P
    if not mem.get("R3") and vetoed != "R3" and "reactor_pressure" not in f["substituted"] and not f["A_RLEVEL_HI"]:
        mem["R3"] = 1
        mem["last"] = "R3"
        return {"rule": "R3", "diagnosis": "sensor_fault", "why": "not RLEVEL_HI", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    # R4: if not TEMP_HIHI and T_avg > 120.36 -> reactor_pressure-50
    if not mem.get("R4") and vetoed != "R4" and f["P_sp"] > 2500.0 and not f["A_TEMP_HIHI"] and f["T_avg"] > 120.36:
        mem["R4"] = 1
        mem["last"] = "R4"
        return {"rule": "R4", "diagnosis": "unknown", "why": "not TEMP_HIHI and T_avg > 120.36", "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": max(2500.0, f["P_sp"] - 50.0)}}}
    # R5: if XL_slope > 1.3255 and n_alarms < 3.5 -> arm_composition
    if not mem.get("R5") and vetoed not in ("R5", "R6") and not f["armed_composition"] and f["XL_slope"] > 1.3255 and f["n_alarms"] < 3.5:
        mem["R5"] = 1
        mem["last"] = "R5"
        return {"rule": "R5", "diagnosis": "unknown", "why": "XL_slope > 1.3255 and n_alarms < 3.5", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    # R6: if not armed_overrides -> arm_composition
    if not mem.get("R6") and vetoed not in ("R5", "R6") and not f["armed_composition"] and not f["armed_overrides"]:
        mem["R6"] = 1
        mem["last"] = "R6"
        return {"rule": "R6", "diagnosis": "unknown", "why": "not armed_overrides", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    # R7: if T_slope > -0.044512 -> arm_both
    if not mem.get("R7") and vetoed not in ("R1", "R2", "R7") and not (f["armed_composition"] and f["armed_overrides"]) and f["T_slope"] > -0.044512:
        mem["R7"] = 1
        mem["last"] = "R7"
        return {"rule": "R7", "diagnosis": "unknown", "why": "T_slope > -0.044512", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    return None
