# synthesised by GP (B5); episode evaluations 19967; generation 8; seed 1
# training-batch success 0.981 (A: 15/15, P: 17/17, Q: 12/12, S: 29/29, SN: 29/31)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if P_ttl < 275.98 -> arm_both
    if not mem.get("R1") and vetoed not in ("R1", "R8") and not (f["armed_composition"] and f["armed_overrides"]) and f["P_ttl"] < 275.98:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "unknown", "why": "P_ttl < 275.98", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    # R2: if T_slope < 0.0023752 -> rate_cut_50
    if not mem.get("R2") and vetoed != "R2" and f["T_slope"] < 0.0023752:
        mem["R2"] = 1
        mem["last"] = "R2"
        return {"rule": "R2", "diagnosis": "unknown", "why": "T_slope < 0.0023752", "action": {"type": "procedure", "name": "rate_cut_50"}}
    # R3: if dP_rs < 5.5438 -> subst_P
    if not mem.get("R3") and vetoed not in ("R3", "R4", "R6") and "reactor_pressure" not in f["substituted"] and f["dP_rs"] < 5.5438:
        mem["R3"] = 1
        mem["last"] = "R3"
        return {"rule": "R3", "diagnosis": "sensor_fault", "why": "dP_rs < 5.5438", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    # R4: if P_slope < 104.5 -> subst_P
    if not mem.get("R4") and vetoed not in ("R3", "R4", "R6") and "reactor_pressure" not in f["substituted"] and f["P_slope"] < 104.5:
        mem["R4"] = 1
        mem["last"] = "R4"
        return {"rule": "R4", "diagnosis": "sensor_fault", "why": "P_slope < 104.5", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    # R5: if dP_rs < -161.51 -> reactor_level-10
    if not mem.get("R5") and vetoed != "R5" and f["RL_sp"] > 50.0 and f["dP_rs"] < -161.51:
        mem["R5"] = 1
        mem["last"] = "R5"
        return {"rule": "R5", "diagnosis": "unknown", "why": "dP_rs < -161.51", "action": {"type": "adjust_setpoints", "changes": {"reactor_level": max(50.0, f["RL_sp"] - 10.0)}}}
    # R6: if XL_slope < 13.014 -> subst_P
    if not mem.get("R6") and vetoed not in ("R3", "R4", "R6") and "reactor_pressure" not in f["substituted"] and f["XL_slope"] < 13.014:
        mem["R6"] = 1
        mem["last"] = "R6"
        return {"rule": "R6", "diagnosis": "sensor_fault", "why": "XL_slope < 13.014", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    # R7: if n_saturated < 0.5 -> subst_T
    if not mem.get("R7") and vetoed != "R7" and "reactor_temperature" not in f["substituted"] and f["n_saturated"] < 0.5:
        mem["R7"] = 1
        mem["last"] = "R7"
        return {"rule": "R7", "diagnosis": "sensor_fault", "why": "n_saturated < 0.5", "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}}
    # R8: if not armed_composition -> arm_both
    if not mem.get("R8") and vetoed not in ("R1", "R8") and not (f["armed_composition"] and f["armed_overrides"]) and not f["armed_composition"]:
        mem["R8"] = 1
        mem["last"] = "R8"
        return {"rule": "R8", "diagnosis": "unknown", "why": "not armed_composition", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    return None
