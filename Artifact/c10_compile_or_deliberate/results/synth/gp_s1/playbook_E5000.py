# synthesised by GP (B5); episode evaluations 4962; generation 1; seed 1
# training-batch success 0.925 (A: 12/12, P: 16/16, Q: 12/12, S: 18/18, SN: 16/22)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if P_ttl < 275.98 -> arm_both
    if not mem.get("R1") and vetoed != "R1" and not (f["armed_composition"] and f["armed_overrides"]) and f["P_ttl"] < 275.98:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "unknown", "why": "P_ttl < 275.98", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    # R2: if P_slope < 135.25 -> subst_P
    if not mem.get("R2") and vetoed != "R2" and "reactor_pressure" not in f["substituted"] and f["P_slope"] < 135.25:
        mem["R2"] = 1
        mem["last"] = "R2"
        return {"rule": "R2", "diagnosis": "sensor_fault", "why": "P_slope < 135.25", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    return None
