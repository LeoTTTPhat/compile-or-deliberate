# synthesised by GP (B5); episode evaluations 1872; generation 0; seed 1
# training-batch success 0.859 (A: 12/12, P: 16/16, Q: 12/12, S: 18/18, SN: 9/20)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if not PRESSURE_HIHI -> arm_both
    if not mem.get("R1") and vetoed != "R1" and not (f["armed_composition"] and f["armed_overrides"]) and not f["A_PRESSURE_HIHI"]:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "unknown", "why": "not PRESSURE_HIHI", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    # R2: if XL_avg > 44.318 -> depressurize_step
    if not mem.get("R2") and vetoed != "R2" and f["XL_avg"] > 44.318:
        mem["R2"] = 1
        mem["last"] = "R2"
        return {"rule": "R2", "diagnosis": "unknown", "why": "XL_avg > 44.318", "action": {"type": "procedure", "name": "depressurize_step"}}
    return None
