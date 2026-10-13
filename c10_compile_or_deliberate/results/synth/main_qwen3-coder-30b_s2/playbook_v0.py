# synthesised by qwen3-coder:30b; tokens 5008; iteration 1; prompt hash 75ea6b0352e43542
# training-batch success 0.359 (A: 12/12, P: 4/16, Q: 0/12, S: 2/18, SN: 10/20)
def playbook(f, mem):
    # Rule 1: Immediate pressure HIHI - initiate depressurization
    if f["A_PRESSURE_HIHI"]:
        return {
            "rule": "PRESSURE_HIHI",
            "diagnosis": "unknown",
            "why": "pressure above limit",
            "action": {"type": "procedure", "name": "depressurize_step"}
        }

    # Rule 2: Immediate temperature HIHI - initiate rate cut and cooling adjustment
    if f["A_TEMP_HIHI"]:
        return {
            "rule": "TEMP_HIHI",
            "diagnosis": "unknown",
            "why": "temperature above limit",
            "action": {"type": "procedure", "name": "rate_cut_50"}
        }

    # Rule 3: Pressure HI with rising trend - adjust setpoint and check for sensor fault
    if f["A_PRESSURE_HI"] and f["P_slope"] > 0:
        # Check if pressure reading is inconsistent with others
        if abs(f["P_resid"]) > 50 or abs(f["dP_rs"]) > 100:
            return {
                "rule": "PRESSURE_SENSOR_FAULT",
                "diagnosis": "sensor_fault",
                "why": "pressure inconsistent with neighbors",
                "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}
            }
        # Else adjust setpoint to reduce pressure
        target = max(2500, f["P_sp"] - 50)
        return {
            "rule": "PRESSURE_HI_ADJUST",
            "diagnosis": "unknown",
            "why": "pressure rising toward limit",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_pressure": target}}
        }

    # Rule 4: Temperature HI with rising trend - adjust setpoint or check sensor
    if f["A_TEMP_HI"] and f["T_slope"] > 0:
        if abs(f["T_resid"]) > 10:
            return {
                "rule": "TEMP_SENSOR_FAULT",
                "diagnosis": "sensor_fault",
                "why": "temperature inconsistent with soft estimate",
                "action": {"type": "substitute_instrument", "measurement": "reactor_temperature", "method": "soft"}
            }
        target = max(115, f["T_sp"] - 2)
        return {
            "rule": "TEMP_HI_ADJUST",
            "diagnosis": "unknown",
            "why": "temperature rising toward limit",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_temperature": target}}
        }

    # Rule 5: Reactor level high - check if it's due to feed or control issue
    if f["A_RLEVEL_HI"]:
        # If level rising and pressure is also high, check for composition upset
        if f["RL_slope"] > 0 and f["P_slope"] > 0:
            return {
                "rule": "RLEVEL_HI_COMPOSITION",
                "diagnosis": "feed_composition_upset",
                "why": "level and pressure rising together",
                "action": {"type": "reconfigure", "components": ["composition_control"]}
            }
        # Else just adjust setpoint
        target = max(50, f["RL_sp"] - 5)
        return {
            "rule": "RLEVEL_HI_ADJUST",
            "diagnosis": "unknown",
            "why": "reactor level high",
            "action": {"type": "adjust_setpoints", "changes": {"reactor_level": target}}
        }

    # Rule 6: Separator level low - check for feed loss or pressure issues
    if f["A_SLEVEL_LO"]:
        # If separator level is dropping and reactor level is rising, likely a feed loss
        if f["SL_slope"] < 0 and f["RL_slope"] > 0:
            return {
                "rule": "SLEVEL_LO_FEEDLOSS",
                "diagnosis": "feed_loss_or_header_pressure",
                "why": "separator low while reactor high",
                "action": {"type": "reconfigure", "components": ["safety_overrides"]}
            }
        # Else just adjust setpoint
        target = min(70, f["SL_sp"] + 5)
        return {
            "rule": "SLEVEL_LO_ADJUST",
            "diagnosis": "unknown",
            "why": "separator level low",
            "action": {"type": "adjust_setpoints", "changes": {"separator_level": target}}
        }

    # Rule 7: Product grade deviation for more than 0.5 hours
    if f["A_GRADE_DEV"] and f["alarm_ages"]["A_GRADE_DEV"] > 0.5:
        return {
            "rule": "GRADE_DEV",
            "diagnosis": "unknown",
            "why": "product grade deviating",
            "action": {"type": "reconfigure", "components": ["composition_control"]}
        }

    # Rule 8: Production rate deviation for more than 1 hour
    if f["A_RATE_DEV"] and f["alarm_ages"]["A_RATE_DEV"] > 1:
        return {
            "rule": "RATE_DEV",
            "diagnosis": "unknown",
            "why": "production rate deviating",
            "action": {"type": "procedure", "name": "rate_cut_25"}
        }

    # Rule 9: Heartbeat and no alarms - check for stuck valves or drifting sensors
    if f["trigger"] == "heartbeat":
        # If any valve is saturated (not moving) for more than 1 hour, consider it stuck
        stuck = [v for v in f["saturated"] if f["alarm_ages"].get(v, 0) > 1]
        if stuck:
            return {
                "rule": "STUCK_VALVE",
                "diagnosis": "valve_stiction_or_stuck",
                "why": "valve not moving",
                "action": {"type": "no_action"}
            }

    # Rule 10: General heartbeat or alarm hold - check for potential drifting sensors
    if f["trigger"] == "heartbeat" or f["trigger"] == "alarm_hold":
        # If reactor pressure is stable but residuals are large, sensor may be drifting
        if abs(f["P_resid"]) > 30 and abs(f["P_slope"]) < 0.1:
            return {
                "rule": "P_SENSOR_DRIFT",
                "diagnosis": "sensor_fault",
                "why": "pressure reading not changing with process",
                "action": {"type": "substitute_instrument", "measurement": "reactor_pressure", "method": "soft"}
            }

    # Default no action
    return None
