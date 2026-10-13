# synthesised by qwen3-coder:30b; tokens 407826; iteration 19; prompt hash 75ea6b0352e43542
# training-batch success 0.737 (A: 12/12, P: 16/17, Q: 12/12, S: 17/23, SN: 16/35)
def playbook(f, mem):
    # Rule 1: Immediate trip limit violation - shutdown
    if f["P_ttl"] < 0.25 or f["T_ttl"] < 0.25 or f["RL_ttl"] < 0.25 or f["SL_ttl"] < 0.25 or f["XL_ttl"] < 0.25:
        return {"rule": "trip_violation", "diagnosis": "unknown", "why": "hard limit near", "action": {"type": "shutdown"}}
    
    # Rule 2: Reactor pressure HIHI - initiate depressurization
    if f["A_PRESSURE_HIHI"]:
        return {"rule": "pressure_hihi", "diagnosis": "unknown", "why": "pressure hihi", "action": {"type": "procedure", "name": "depressurize_step"}}
    
    # Rule 3: Reactor temperature HIHI - reduce production rate
    if f["A_TEMP_HIHI"]:
        return {"rule": "temp_hihi", "diagnosis": "unknown", "why": "temperature hihi", "action": {"type": "procedure", "name": "rate_cut_25"}}
    
    # Rule 4: Reactor pressure sensor fault - substitute with soft sensor
    if f["P_resid"] > 400 and "reactor_pressure" not in f["substituted"]:
        return {"rule": "pressure_sensor_fault", "diagnosis": "sensor_fault", "why": "pressure reading inconsistent", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    
    # Rule 5: Reactor temperature sensor fault - substitute with soft sensor
    if f["T_resid"] > 20 and "reactor_temperature" not in f["substituted"]:
        return {"rule": "temperature_sensor_fault", "diagnosis": "sensor_fault", "why": "temperature reading inconsistent", "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}}
    
    # Rule 6: Separator temperature sensor fault - substitute with soft sensor
    if f["ST"] > 100 and "separator_temperature" not in f["substituted"]:
        return {"rule": "separator_temp_sensor_fault", "diagnosis": "sensor_fault", "why": "separator temp reading inconsistent", "action": {"type": "substitute_instrument", "measurement": "separator_temperature", "method": "soft"}}
    
    # Rule 7: Stripper pressure sensor fault - substitute with soft sensor
    if f["XP"] > 100 and "stripper_pressure" not in f["substituted"]:
        return {"rule": "stripper_pressure_sensor_fault", "diagnosis": "sensor_fault", "why": "stripper pressure reading inconsistent", "action": {"type": "substitute_instrument", "measurement": "stripper_pressure", "method": "soft"}}
    
    # Rule 8: Reactor level HI - adjust setpoint to lower
    if f["A_RLEVEL_HI"] and f["RL_sp"] > 65:
        return {"rule": "reactor_level_hi", "diagnosis": "unknown", "why": "reactor level high", "action": {"type": "adjust_setpoints", "changes": {"reactor_level": max(50, f["RL_sp"] - 5)}}}
    
    # Rule 9: Reactor level LO - adjust setpoint to higher
    if f["A_RLEVEL_LO"] and f["RL_sp"] < 75:
        return {"rule": "reactor_level_lo", "diagnosis": "unknown", "why": "reactor level low", "action": {"type": "adjust_setpoints", "changes": {"reactor_level": min(85, f["RL_sp"] + 5)}}}
    
    # Rule 10: Separator level HI - adjust setpoint to lower
    if f["A_SLEVEL_HI"] and f["SL_sp"] > 60:
        return {"rule": "separator_level_hi", "diagnosis": "unknown", "why": "separator level high", "action": {"type": "adjust_setpoints", "changes": {"separator_level": max(30, f["SL_sp"] - 5)}}}
    
    # Rule 11: Separator level LO - adjust setpoint to higher
    if f["A_SLEVEL_LO"] and f["SL_sp"] < 40:
        return {"rule": "separator_level_lo", "diagnosis": "unknown", "why": "separator level low", "action": {"type": "adjust_setpoints", "changes": {"separator_level": min(70, f["SL_sp"] + 5)}}}
    
    # Rule 12: Stripper level HI - adjust setpoint to lower
    if f["A_XLEVEL_HI"] and f["XL_sp"] > 60:
        return {"rule": "stripper_level_hi", "diagnosis": "unknown", "why": "stripper level high", "action": {"type": "adjust_setpoints", "changes": {"stripper_level": max(30, f["XL_sp"] - 5)}}}
    
    # Rule 13: Stripper level LO - adjust setpoint to higher
    if f["A_XLEVEL_LO"] and f["XL_sp"] < 60:
        return {"rule": "stripper_level_lo", "diagnosis": "unknown", "why": "stripper level low", "action": {"type": "adjust_setpoints", "changes": {"stripper_level": min(70, f["XL_sp"] + 5)}}}
    
    # Rule 14: Product grade deviation - arm composition control
    if f["A_GRADE_DEV"] and not f["armed_composition"]:
        return {"rule": "grade_deviation", "diagnosis": "feed_composition_upset", "why": "product grade off target", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    
    # Rule 15: Production rate deviation - arm safety overrides if not already armed
    if f["A_RATE_DEV"] and not f["armed_overrides"]:
        return {"rule": "rate_deviation", "diagnosis": "unknown", "why": "production rate off target", "action": {"type": "reconfigure", "components": ["safety_overrides"]}}
    
    # Rule 16: High pressure alarm - initiate rate cut if not already doing so
    if f["A_PRESSURE_HI"] and f["cmd_production"] > 20:
        return {"rule": "pressure_high", "diagnosis": "unknown", "why": "pressure high", "action": {"type": "procedure", "name": "rate_cut_25"}}
    
    # Rule 17: High temperature alarm - initiate rate cut if not already doing so
    if f["A_TEMP_HI"] and f["cmd_production"] > 20:
        return {"rule": "temperature_high", "diagnosis": "unknown", "why": "temperature high", "action": {"type": "procedure", "name": "rate_cut_25"}}
    
    # Rule 18: Low temperature alarm - adjust setpoint to higher if not already doing so
    if f["A_TEMP_LO"] and f["T_sp"] < 125:
        return {"rule": "temperature_low", "diagnosis": "unknown", "why": "temperature low", "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": min(128, f["T_sp"] + 2)}}}
    
    # Rule 19: Low pressure alarm - adjust setpoint to higher if not already doing so
    if f["A_PRESSURE_LO"] and f["P_sp"] < 2700:
        return {"rule": "pressure_low", "diagnosis": "unknown", "why": "pressure low", "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": min(2895, f["P_sp"] + 50)}}}
    
    # Rule 20: Pressure trending toward HIHI - initiate rate cut if not already doing so
    if (f["P_slope"] > 10 and f["P_ttl"] < 2.0) and f["cmd_production"] > 20:
        return {"rule": "pressure_trending_hihi", "diagnosis": "unknown", "why": "pressure trending high", "action": {"type": "procedure", "name": "rate_cut_25"}}
    
    # Rule 21: Temperature trending toward HIHI - initiate rate cut if not already doing so
    if (f["T_slope"] > 0.5 and f["T_ttl"] < 2.0) and f["cmd_production"] > 20:
        return {"rule": "temp_trending_hihi", "diagnosis": "unknown", "why": "temperature trending high", "action": {"type": "procedure", "name": "rate_cut_25"}}
    
    # Rule 22: Reactor pressure drift - substitute with soft sensor if no other alarms and pressure slope is large
    # Only substitute if there are no active alarms, the pressure is not frozen (slope = 0) and not already substituted
    if (f["P_slope"] < -2.5 or f["P_slope"] > 10) and not f["alarms"] and "reactor_pressure" not in f["substituted"] and abs(f["P_slope"]) > 0.1:
        return {"rule": "pressure_drift", "diagnosis": "sensor_fault", "why": "pressure drift detected", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    
    # Rule 23: Reactor temperature drift - substitute with soft sensor if no other alarms and temperature slope is large
    if (f["T_slope"] < -1 or f["T_slope"] > 1) and not f["alarms"] and "reactor_temperature" not in f["substituted"] and abs(f["T_slope"]) > 0.1:
        return {"rule": "temperature_drift", "diagnosis": "sensor_fault", "why": "temperature drift detected", "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}}
    
    # Rule 24: Feed composition upset - arm composition control if not already armed
    if f["yA_dev_nom"] < -5 and not f["armed_composition"]:
        return {"rule": "feed_composition_upset", "diagnosis": "feed_composition_upset", "why": "feed composition upset", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    
    # Rule 25: Reactor pressure frozen - substitute with soft sensor if not already substituted
    if f["P_slope"] == 0 and abs(f["P_resid"]) > 10 and "reactor_pressure" not in f["substituted"]:
        return {"rule": "pressure_frozen", "diagnosis": "sensor_fault", "why": "pressure reading frozen", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    
    # Rule 26: Reactor temperature frozen - substitute with soft sensor if not already substituted
    if f["T_slope"] == 0 and abs(f["T_resid"]) > 10 and "reactor_temperature" not in f["substituted"]:
        return {"rule": "temperature_frozen", "diagnosis": "sensor_fault", "why": "temperature reading frozen", "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}}
    
    # Rule 27: Reactor pressure trending downward toward low limit - adjust setpoint up
    if f["P_slope"] < -10 and f["P_ttl"] < 3.0:
        return {"rule": "pressure_trending_low", "diagnosis": "unknown", "why": "pressure trending low", "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": min(2895, f["P_sp"] + 30)}}}
    
    # Rule 28: Reactor temperature trending upward toward high limit - adjust setpoint down
    if f["T_slope"] > 1 and f["T_ttl"] < 3.0:
        return {"rule": "temp_trending_high", "diagnosis": "unknown", "why": "temperature trending high", "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": max(115, f["T_sp"] - 2)}}}
    
    # Rule 29: Feed flows A/C ratio change detected - arm composition control if not already armed
    if f["yA_dev_nom"] < -3 and not f["armed_composition"]:
        return {"rule": "feed_ratio_change", "diagnosis": "feed_composition_upset", "why": "feed ratio changed", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    
    # Rule 30: Reactor temperature trending with no alarm - check for drift and substitute if necessary
    if (f["T_slope"] < -1 or f["T_slope"] > 1) and not f["alarms"] and "reactor_temperature" not in f["substituted"] and abs(f["T_slope"]) > 0.5:
        return {"rule": "temperature_trending_drift", "diagnosis": "sensor_fault", "why": "temperature trending drift", "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}}
    
    # Rule 31: Reactor pressure trending with no alarm - check for drift and substitute if necessary
    if (f["P_slope"] < -2.5 or f["P_slope"] > 10) and not f["alarms"] and "reactor_pressure" not in f["substituted"] and abs(f["P_slope"]) > 2:
        return {"rule": "pressure_trending_drift", "diagnosis": "sensor_fault", "why": "pressure trending drift", "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}}
    
    # Rule 32: Feed loss or header pressure loss - check for low feed flow and arm composition control
    if f["FA"] < 10 and not f["armed_composition"]:
        return {"rule": "feed_loss_or_header_pressure", "diagnosis": "feed_loss_or_header_pressure", "why": "low feed flow detected", "action": {"type": "reconfigure", "components": ["composition_control"]}}
    
    # Rule 33: No alarm or trend - no action
    return None
