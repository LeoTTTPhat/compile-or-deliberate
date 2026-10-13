# synthesised by qwen2.5-coder:7b; tokens 105631; iteration 1; prompt hash 75ea6b0352e43542
# training-batch success 0.308 (A: 12/12, P: 4/16, Q: 0/12, S: 2/18, SN: 6/20)
def playbook(f, mem):
    if f["trigger"] == "alarm_hold":
        return None

    # Reactor pressure is approaching trip limit
    if f["P_ttl"] < 1:
        return {
            "rule": "P_TTL",
            "diagnosis": "reactor_pressure",
            "why": "Pressure is approaching trip limit",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": f["P_sp"] - 100}}
        }

    # Reactor temperature is approaching trip limit
    if f["T_ttl"] < 1:
        return {
            "rule": "T_TTL",
            "diagnosis": "reactor_temperature",
            "why": "Temperature is approaching trip limit",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": f["T_sp"] - 3}}
        }

    # Reactor level is out of range
    if f["RL"] < 45 or f["RL"] > 92:
        return {
            "rule": "RL_OutOfRange",
            "diagnosis": "reactor_level",
            "why": "Level is out of range",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_level": f["RL_sp"]}}
        }

    # Separator level is out of range
    if f["SL"] < 15 or f["SL"] > 85:
        return {
            "rule": "SL_OutOfRange",
            "diagnosis": "separator_level",
            "why": "Level is out of range",
            "action": {"type": "adjust_setpoints", "changes": {"separator_level": f["SL_sp"]}}
        }

    # Stripper level is out of range
    if f["XL"] < 15 or f["XL"] > 85:
        return {
            "rule": "XL_OutOfRange",
            "diagnosis": "stripper_level",
            "why": "Level is out of range",
            "action": {"type": "adjust_setpoints", "changes": {"stripper_level": f["XL_sp"]}}
        }

    # Reactor pressure is above nominal
    if f["P"] > 2705:
        return {
            "rule": "P_Hi",
            "diagnosis": "reactor_pressure",
            "why": "Pressure is above nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": 2705}}
        }

    # Reactor temperature is above nominal
    if f["T"] > 120.4:
        return {
            "rule": "T_Hi",
            "diagnosis": "reactor_temperature",
            "why": "Temperature is above nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": 120.4}}
        }

    # Reactor pressure is below nominal
    if f["P"] < 2705:
        return {
            "rule": "P_Lo",
            "diagnosis": "reactor_pressure",
            "why": "Pressure is below nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": 2705}}
        }

    # Reactor temperature is below nominal
    if f["T"] < 120.4:
        return {
            "rule": "T_Lo",
            "diagnosis": "reactor_temperature",
            "why": "Temperature is below nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": 120.4}}
        }

    # Reactor level is below nominal
    if f["RL"] < 75:
        return {
            "rule": "RL_Lo",
            "diagnosis": "reactor_level",
            "why": "Level is below nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_level": 75}}
        }

    # Separator level is below nominal
    if f["SL"] < 50:
        return {
            "rule": "SL_Lo",
            "diagnosis": "separator_level",
            "why": "Level is below nominal",
            "action": {"type": "adjust_setpoints", "changes": {"separator_level": 50}}
        }

    # Stripper level is below nominal
    if f["XL"] < 50:
        return {
            "rule": "XL_Lo",
            "diagnosis": "stripper_level",
            "why": "Level is below nominal",
            "action": {"type": "adjust_setpoints", "changes": {"stripper_level": 50}}
        }

    # Reactor pressure is above hard limit
    if f["P"] > 3000:
        return {
            "rule": "P_HiHi",
            "diagnosis": "reactor_pressure",
            "why": "Pressure is above hard limit",
            "action": {"type": "shutdown"}
        }

    # Reactor temperature is above hard limit
    if f["T"] > 175:
        return {
            "rule": "T_HiHi",
            "diagnosis": "reactor_temperature",
            "why": "Temperature is above hard limit",
            "action": {"type": "shutdown"}
        }

    # Reactor level is above hard limit
    if f["RL"] > 100:
        return {
            "rule": "RL_HiHi",
            "diagnosis": "reactor_level",
            "why": "Level is above hard limit",
            "action": {"type": "shutdown"}
        }

    # Separator level is above hard limit
    if f["SL"] > 100:
        return {
            "rule": "SL_HiHi",
            "diagnosis": "separator_level",
            "why": "Level is above hard limit",
            "action": {"type": "shutdown"}
        }

    # Stripper level is above hard limit
    if f["XL"] > 100:
        return {
            "rule": "XL_HiHi",
            "diagnosis": "stripper_level",
            "why": "Level is above hard limit",
            "action": {"type": "shutdown"}
        }

    # Reactor pressure is below hard limit
    if f["P"] < 0:
        return {
            "rule": "P_LoLo",
            "diagnosis": "reactor_pressure",
            "why": "Pressure is below hard limit",
            "action": {"type": "shutdown"}
        }

    # Reactor temperature is below hard limit
    if f["T"] < 0:
        return {
            "rule": "T_LoLo",
            "diagnosis": "reactor_temperature",
            "why": "Temperature is below hard limit",
            "action": {"type": "shutdown"}
        }

    # Reactor level is below hard limit
    if f["RL"] < 0:
        return {
            "rule": "RL_LoLo",
            "diagnosis": "reactor_level",
            "why": "Level is below hard limit",
            "action": {"type": "shutdown"}
        }

    # Separator level is below hard limit
    if f["SL"] < 0:
        return {
            "rule": "SL_LoLo",
            "diagnosis": "separator_level",
            "why": "Level is below hard limit",
            "action": {"type": "shutdown"}
        }

    # Stripper level is below hard limit
    if f["XL"] < 0:
        return {
            "rule": "XL_LoLo",
            "diagnosis": "stripper_level",
            "why": "Level is below hard limit",
            "action": {"type": "shutdown"}
        }

    # Reactor pressure is outside normal range
    if f["P"] < 2500 or f["P"] > 2895:
        return {
            "rule": "P_OutOfNom",
            "diagnosis": "reactor_pressure",
            "why": "Pressure is outside normal range",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": 2705}}
        }

    # Reactor temperature is outside normal range
    if f["T"] < 115 or f["T"] > 128:
        return {
            "rule": "T_OutOfNom",
            "diagnosis": "reactor_temperature",
            "why": "Temperature is outside normal range",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": 120.4}}
        }

    # Reactor level is outside normal range
    if f["RL"] < 50 or f["RL"] > 85:
        return {
            "rule": "RL_OutOfNom",
            "diagnosis": "reactor_level",
            "why": "Level is outside normal range",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_level": 75}}
        }

    # Separator level is outside normal range
    if f["SL"] < 30 or f["SL"] > 70:
        return {
            "rule": "SL_OutOfNom",
            "diagnosis": "separator_level",
            "why": "Level is outside normal range",
            "action": {"type": "adjust_setpoints", "changes": {"separator_level": 50}}
        }

    # Stripper level is outside normal range
    if f["XL"] < 30 or f["XL"] > 70:
        return {
            "rule": "XL_OutOfNom",
            "diagnosis": "stripper_level",
            "why": "Level is outside normal range",
            "action": {"type": "adjust_setpoints", "changes": {"stripper_level": 50}}
        }

    # Reactor pressure is above soft sensor estimate
    if f["P_resid"] > 0:
        return {
            "rule": "P_SoftHi",
            "diagnosis": "reactor_pressure",
            "why": "Pressure is above soft sensor estimate",
            "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}
        }

    # Reactor temperature is above soft sensor estimate
    if f["T_resid"] > 0:
        return {
            "rule": "T_SoftHi",
            "diagnosis": "reactor_temperature",
            "why": "Temperature is above soft sensor estimate",
            "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}
        }

    # Reactor pressure is below soft sensor estimate
    if f["P_resid"] < 0:
        return {
            "rule": "P_SoftLo",
            "diagnosis": "reactor_pressure",
            "why": "Pressure is below soft sensor estimate",
            "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}
        }

    # Reactor temperature is below soft sensor estimate
    if f["T_resid"] < 0:
        return {
            "rule": "T_SoftLo",
            "diagnosis": "reactor_temperature",
            "why": "Temperature is below soft sensor estimate",
            "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}
        }

    # Reactor pressure is below nominal
    if f["P"] < 2705:
        return {
            "rule": "P_Lo",
            "diagnosis": "reactor_pressure",
            "why": "Pressure is below nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": 2705}}
        }

    # Reactor temperature is below nominal
    if f["T"] < 120.4:
        return {
            "rule": "T_Lo",
            "diagnosis": "reactor_temperature",
            "why": "Temperature is below nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": 120.4}}
        }

    # Reactor level is below nominal
    if f["RL"] < 75:
        return {
            "rule": "RL_Lo",
            "diagnosis": "reactor_level",
            "why": "Level is below nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_level": 75}}
        }

    # Separator level is below nominal
    if f["SL"] < 50:
        return {
            "rule": "SL_Lo",
            "diagnosis": "separator_level",
            "why": "Level is below nominal",
            "action": {"type": "adjust_setpoints", "changes": {"separator_level": 50}}
        }

    # Stripper level is below nominal
    if f["XL"] < 50:
        return {
            "rule": "XL_Lo",
            "diagnosis": "stripper_level",
            "why": "Level is below nominal",
            "action": {"type": "adjust_setpoints", "changes": {"stripper_level": 50}}
        }

    # Reactor pressure is above nominal
    if f["P"] > 2705:
        return {
            "rule": "P_Hi",
            "diagnosis": "reactor_pressure",
            "why": "Pressure is above nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": 2705}}
        }

    # Reactor temperature is above nominal
    if f["T"] > 120.4:
        return {
            "rule": "T_Hi",
            "diagnosis": "reactor_temperature",
            "why": "Temperature is above nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": 120.4}}
        }

    # Reactor level is above nominal
    if f["RL"] > 75:
        return {
            "rule": "RL_Hi",
            "diagnosis": "reactor_level",
            "why": "Level is above nominal",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_level": 75}}
        }

    # Separator level is above nominal
    if f["SL"] > 50:
        return {
            "rule": "SL_Hi",
            "diagnosis": "separator_level",
            "why": "Level is above nominal",
            "action": {"type": "adjust_setpoints", "changes": {"separator_level": 50}}
        }

    # Stripper level is above nominal
    if f["XL"] > 50:
        return {
            "rule": "XL_Hi",
            "diagnosis": "stripper_level",
            "why": "Level is above nominal",
            "action": {"type": "adjust_setpoints", "changes": {"stripper_level": 50}}
        }

    # No abnormal situation detected
    return None
