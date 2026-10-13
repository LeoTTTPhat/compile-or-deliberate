# synthesised by qwen3-coder:30b; tokens 51629; iteration 3; prompt hash 75ea6b0352e43542
# training-batch success 0.989 (A: 12/12, P: 20/20, Q: 12/12, S: 25/25, SN: 20/21)
def playbook(f, mem):
    
    # Rule 14: High pressure approaching trip - arm safety override
    if f["P_margin"] < 200 and not f["armed_overrides"]:
        return {"rule": "PRESSURE_APPROACHING_TRIP", "diagnosis": "unknown", "why": "pressure near limit", "action": {"type": "reconfigure", "components": ["safety_overrides"]}}
    # Rule 1: Immediate trip limit violation - shutdown
    if f["P_ttl"] < 0.25 or f["T_ttl"] < 0.25 or f["RL_ttl"] < 0.25 or f["SL_ttl"] < 0.25 or f["XL_ttl"] < 0.25:
        return {"rule": "TRIP_VIOLATION", "diagnosis": "unknown", "why": "hard limit imminent", "action": {"type": "shutdown"}}
    
    # Rule 2: Pressure HIHI alarm - immediate action to reduce pressure
    if f["A_PRESSURE_HIHI"]:
        return {"rule": "PRESSURE_HIHI", "diagnosis": "sensor_fault", "why": "pressure too high", "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": min(2895, f["P_sp"] - 75)}}}
    
    # Rule 3: Temperature HIHI alarm - immediate action to reduce temperature
    if f["A_TEMP_HIHI"]:
        return {"rule": "TEMP_HIHI", "diagnosis": "sensor_fault", "why": "temperature too high", "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": min(128, f["T_sp"] - 3)}}}
    
    # Rule 4: Reactor level HI alarm with increasing trend - adjust setpoint
    if f["A_RLEVEL_HI"] and f["RL_slope"] > 0:
        return {"rule": "RLEVEL_HI", "diagnosis": "unknown", "why": "level rising", "action": {"type": "adjust_setpoints", "changes": {"reactor_level": max(75, f["RL_sp"] - 5)}}}
    
    # Rule 5: Separator level LO alarm with decreasing trend - adjust setpoint
    if f["A_SLEVEL_LO"] and f["SL_slope"] < 0:
        return {"rule": "SLEVEL_LO", "diagnosis": "unknown", "why": "level falling", "action": {"type": "adjust_setpoints", "changes": {"separator_level": min(50, f["SL_sp"] + 5)}}}
    
    # Rule 6: Product grade deviation - use composition control
    if f["A_GRADE_DEV"] and not f["armed_composition"]:
        return {"rule": "GRADE_DEV", "diagnosis": "feed_composition_upset", "why": "grade off target", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    
    # Rule 7: Pressure HI alarm - reduce pressure setpoint
    if f["A_PRESSURE_HI"]:
        return {"rule": "PRESSURE_HI", "diagnosis": "sensor_fault", "why": "pressure high", "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": min(2895, f["P_sp"] - 20)}}}
    
    # Rule 8: Temperature HI alarm - reduce temperature setpoint
    if f["A_TEMP_HI"]:
        return {"rule": "TEMP_HI", "diagnosis": "sensor_fault", "why": "temperature high", "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": min(128, f["T_sp"] - 1)}}}
    
    # Rule 9: Reactor level LO alarm with decreasing trend - adjust setpoint
    if f["A_RLEVEL_LO"] and f["RL_slope"] < 0:
        return {"rule": "RLEVEL_LO", "diagnosis": "unknown", "why": "level falling", "action": {"type": "adjust_setpoints", "changes": {"reactor_level": min(85, f["RL_sp"] + 5)}}}
    
    # Rule 10: Separator level HI alarm with increasing trend - adjust setpoint
    if f["A_SLEVEL_HI"] and f["SL_slope"] > 0:
        return {"rule": "SLEVEL_HI", "diagnosis": "unknown", "why": "level rising", "action": {"type": "adjust_setpoints", "changes": {"separator_level": max(30, f["SL_sp"] - 5)}}}
    
    # Rule 11: Product rate deviation - reduce production rate
    if f["A_RATE_DEV"]:
        return {"rule": "RATE_DEV", "diagnosis": "unknown", "why": "rate off target", "action": {"type": "adjust_setpoints", "changes": {"production_rate": max(11, f["PROD_sp"] - 2)}}}
    
    # Rule 12: Reactor pressure sensor fault - substitute with soft sensor
    if ("reactor_pressure" in f["substituted"] or abs(f["P_resid"]) > 50) and not mem.get("pressure_sensor_substituted"):
        return {"rule": "PRESSURE_SENSOR_FAULT", "diagnosis": "sensor_fault", "why": "pressure sensor faulty", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    
    # Rule 13: Reactor temperature sensor fault - substitute with soft sensor
    if ("reactor_temperature" in f["substituted"] or abs(f["T_resid"]) > 5) and not mem.get("temperature_sensor_substituted"):
        return {"rule": "TEMP_SENSOR_FAULT", "diagnosis": "sensor_fault", "why": "temperature sensor faulty", "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}}
    
    # Rule 15: Reactor cooling water valve stuck - check for saturation
    if "mv_r_cooling_water" in f["saturated"] and not mem.get("cooling_water_reconfigured"):
        return {"rule": "COOLING_WATER_VALVE_STUCK", "diagnosis": "valve_stiction_or_stuck", "why": "cooling valve stuck", "action": {"type": "procedure", "name": "min_stable_park"}}
    
    # Rule 16: Feed composition upset detected by large deviation in yA and increasing pressure slope
    if f["yA_dev_nom"] < -10 and f["P_slope"] > 20 and not mem.get("composition_control_armed"):
        return {"rule": "FEED_COMPOSITION_UPSET", "diagnosis": "feed_composition_upset", "why": "A/C ratio upset", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    
    # Rule 17: Feed loss detected by low A feed and increasing pressure
    if f["FA"] < 20 and f["P_slope"] > 30 and not mem.get("safety_overrides_armed"):
        return {"rule": "FEED_LOSS", "diagnosis": "feed_loss_or_header_pressure", "why": "A feed loss", "action": {"type": "reconfigure", "components": ["safety_overrides"]}}
    
    # Rule 18: Temperature drift detected by large T_resid and stable pressure
    if abs(f["T_resid"]) > 1.5 and f["P_slope"] < 5 and not mem.get("temperature_sensor_substituted"):
        return {"rule": "TEMPERATURE_DRIFT", "diagnosis": "sensor_fault", "why": "temperature drift", "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}}
    
    # Rule 19: No alarms, stable conditions - no action
    return {"rule": "NO_ACTION", "diagnosis": "unknown", "why": "normal operation", "action": {"type": "no_action"}}
