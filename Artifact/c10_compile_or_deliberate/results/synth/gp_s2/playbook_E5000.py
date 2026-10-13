# synthesised by GP (B5); episode evaluations 4962; generation 1; seed 2
# training-batch success 0.805 (A: 12/12, P: 16/16, Q: 12/12, S: 18/18, SN: 8/24)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if not GRADE_DEV -> min_stable_park
    if not mem.get("R1") and vetoed != "R1" and not f["A_GRADE_DEV"]:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "unknown", "why": "not GRADE_DEV", "action": {"type": "procedure", "name": "min_stable_park"}}
    # R2: if P_resid < 30.778 -> arm_composition
    if not mem.get("R2") and vetoed != "R2" and not f["armed_composition"] and f["P_resid"] < 30.778:
        mem["R2"] = 1
        mem["last"] = "R2"
        return {"rule": "R2", "diagnosis": "unknown", "why": "P_resid < 30.778", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    return None
