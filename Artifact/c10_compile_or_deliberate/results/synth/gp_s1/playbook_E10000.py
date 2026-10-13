# synthesised by GP (B5); episode evaluations 9921; generation 4; seed 1
# training-batch success 0.900 (A: 12/13, P: 16/16, Q: 12/12, S: 13/21, SN: 28/28)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if P_ttl > 275.98 -> arm_both
    if not mem.get("R1") and vetoed != "R1" and not (f["armed_composition"] and f["armed_overrides"]) and f["P_ttl"] > 275.98:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "unknown", "why": "P_ttl > 275.98", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    # R2: if P_slope < 135.25 -> subst_P
    if not mem.get("R2") and vetoed != "R2" and "reactor_pressure" not in f["substituted"] and f["P_slope"] < 135.25:
        mem["R2"] = 1
        mem["last"] = "R2"
        return {"rule": "R2", "diagnosis": "sensor_fault", "why": "P_slope < 135.25", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    # R3: if n_saturated < 1.5 -> subst_T
    if not mem.get("R3") and vetoed != "R3" and "reactor_temperature" not in f["substituted"] and f["n_saturated"] < 1.5:
        mem["R3"] = 1
        mem["last"] = "R3"
        return {"rule": "R3", "diagnosis": "sensor_fault", "why": "n_saturated < 1.5", "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}}
    # R4: if dP_rs < -161.51 -> reactor_level-10
    if not mem.get("R4") and vetoed != "R4" and f["RL_sp"] > 50.0 and f["dP_rs"] < -161.51:
        mem["R4"] = 1
        mem["last"] = "R4"
        return {"rule": "R4", "diagnosis": "unknown", "why": "dP_rs < -161.51", "action": {"type": "adjust_setpoints", "changes": {"reactor_level": max(50.0, f["RL_sp"] - 10.0)}}}
    return None
