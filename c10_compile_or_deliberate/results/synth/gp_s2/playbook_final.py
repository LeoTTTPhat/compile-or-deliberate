# synthesised by GP (B5); episode evaluations 19959; generation 9; seed 2
# training-batch success 0.746 (A: 13/13, P: 17/17, Q: 12/12, S: 21/21, SN: 22/51)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if P_ttl < 5.8432 and XL_slope > 12.326 -> reactor_temperature+3
    if not mem.get("R1") and vetoed != "R1" and f["T_sp"] < 128.0 and f["P_ttl"] < 5.8432 and f["XL_slope"] > 12.326:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "unknown", "why": "P_ttl < 5.8432 and XL_slope > 12.326", "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": min(128.0, f["T_sp"] + 3.0)}}}
    # R2: if P_resid < 30.778 -> arm_both
    if not mem.get("R2") and vetoed != "R2" and not (f["armed_composition"] and f["armed_overrides"]) and f["P_resid"] < 30.778:
        mem["R2"] = 1
        mem["last"] = "R2"
        return {"rule": "R2", "diagnosis": "unknown", "why": "P_resid < 30.778", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    # R3: if T_avg > 120.4 -> arm_composition
    if not mem.get("R3") and vetoed != "R3" and not f["armed_composition"] and f["T_avg"] > 120.4:
        mem["R3"] = 1
        mem["last"] = "R3"
        return {"rule": "R3", "diagnosis": "unknown", "why": "T_avg > 120.4", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    # R4: if G_cmd_err > -8.432 -> min_stable_park
    if not mem.get("R4") and vetoed != "R4" and f["G_cmd_err"] > -8.432:
        mem["R4"] = 1
        mem["last"] = "R4"
        return {"rule": "R4", "diagnosis": "unknown", "why": "G_cmd_err > -8.432", "action": {"type": "procedure", "name": "min_stable_park"}}
    # R5: if T_avg < 120.43 -> subst_T
    if not mem.get("R5") and vetoed != "R5" and "reactor_temperature" not in f["substituted"] and f["T_avg"] < 120.43:
        mem["R5"] = 1
        mem["last"] = "R5"
        return {"rule": "R5", "diagnosis": "sensor_fault", "why": "T_avg < 120.43", "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}}
    return None
