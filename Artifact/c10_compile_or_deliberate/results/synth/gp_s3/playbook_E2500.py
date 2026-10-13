# synthesised by GP (B5); episode evaluations 2496; generation 0; seed 3
# training-batch success 0.808 (A: 12/12, P: 16/16, Q: 12/12, S: 18/18, SN: 5/20)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if P_resid < -0.043017 -> arm_both
    if not mem.get("R1") and vetoed != "R1" and not (f["armed_composition"] and f["armed_overrides"]) and f["P_resid"] < -0.043017:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "unknown", "why": "P_resid < -0.043017", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    return None
