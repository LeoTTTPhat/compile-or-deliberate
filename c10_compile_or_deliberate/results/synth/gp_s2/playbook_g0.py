# synthesised by GP (B5); episode evaluations 1872; generation 0; seed 2
# training-batch success 0.808 (A: 12/12, P: 16/16, Q: 12/12, S: 18/18, SN: 5/20)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if GRADE_DEV -> reactor_pressure-100
    if not mem.get("R1") and vetoed != "R1" and f["P_sp"] > 2500.0 and f["A_GRADE_DEV"]:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "unknown", "why": "GRADE_DEV", "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": max(2500.0, f["P_sp"] - 100.0)}}}
    # R2: if yA_dev_nom > -14.8 -> arm_both
    if not mem.get("R2") and vetoed != "R2" and not (f["armed_composition"] and f["armed_overrides"]) and f["yA_dev_nom"] > -14.8:
        mem["R2"] = 1
        mem["last"] = "R2"
        return {"rule": "R2", "diagnosis": "unknown", "why": "yA_dev_nom > -14.8", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    # R3: if armed_overrides -> rate_cut_50
    if not mem.get("R3") and vetoed != "R3" and f["armed_overrides"]:
        mem["R3"] = 1
        mem["last"] = "R3"
        return {"rule": "R3", "diagnosis": "unknown", "why": "armed_overrides", "action": {"type": "procedure", "name": "rate_cut_50"}}
    # R4: if SL_avg > 51.174 and mv_a_feed > 15.731 -> arm_overrides
    if not mem.get("R4") and vetoed != "R4" and not f["armed_overrides"] and f["SL_avg"] > 51.174 and f["mv_a_feed"] > 15.731:
        mem["R4"] = 1
        mem["last"] = "R4"
        return {"rule": "R4", "diagnosis": "unknown", "why": "SL_avg > 51.174 and mv_a_feed > 15.731", "action": {"type": "reconfigure", "components": ["safety_overrides"]}}
    return None
