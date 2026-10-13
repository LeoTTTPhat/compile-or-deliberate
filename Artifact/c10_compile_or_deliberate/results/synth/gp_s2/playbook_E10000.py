# synthesised by GP (B5); episode evaluations 9952; generation 4; seed 2
# training-batch success 0.768 (A: 13/13, P: 17/17, Q: 12/12, S: 21/21, SN: 10/32)
def playbook(f, mem):
    # each rule proposes its action at most once per episode; a vetoed proposal is released
    vetoed = ""
    if f.get("trigger") == "revision":
        vetoed = mem.get("last", "")
        mem[vetoed] = 0
    # R1: if PRESSURE_HI and n_alarms < 1.5 -> subst_T
    if not mem.get("R1") and vetoed != "R1" and "reactor_temperature" not in f["substituted"] and f["A_PRESSURE_HI"] and f["n_alarms"] < 1.5:
        mem["R1"] = 1
        mem["last"] = "R1"
        return {"rule": "R1", "diagnosis": "sensor_fault", "why": "PRESSURE_HI and n_alarms < 1.5", "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}}
    # R2: if yA_dev_nom > -14.8 -> arm_both
    if not mem.get("R2") and vetoed != "R2" and not (f["armed_composition"] and f["armed_overrides"]) and f["yA_dev_nom"] > -14.8:
        mem["R2"] = 1
        mem["last"] = "R2"
        return {"rule": "R2", "diagnosis": "unknown", "why": "yA_dev_nom > -14.8", "action": {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}}
    return None
