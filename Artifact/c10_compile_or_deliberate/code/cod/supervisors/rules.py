"""Hand-written, zero-token rule-table supervisors (non-LLM baselines).

B2a `AlarmResponseTable`: Alhazmi's scripted supervisor, reconstructed from the paper's
description ("shed production on critical pressure, activate the standing protections on
sustained alarms, adjust the corresponding setpoint on level alarms"). Generic practice; it
forms no diagnosis.

B2b `ExpertTable`: what a competent process/control engineer would write with explicit
effort: B2a plus quality-deviation handling and instrument-consistency checks (analytical
redundancy) that take a lying transmitter out of service. It represents the human-
engineering ceiling of a rule table built without the test suite.
Both tables were written before any LLM experiment and are not tuned on test scenarios.
"""
from __future__ import annotations

from ..actions import NO_ACTION, Action
from ..runner import Decision


class _Base:
    name = "base"

    def episode_start(self, plant, scenario, seed):
        self.state = {}

    def revise(self, plant, feats, rejected, reasons, history):
        return None

    def episode_end(self, plant, outcome):
        pass


def _sp(field, value):
    return Action("adjust_setpoints", changes=((field, float(value)),))


class AlarmResponseTable(_Base):
    name = "B2a_alarm_table"

    def decide(self, plant, f, trigger, history):
        A = set(f["alarms"])
        armed_c, armed_o = f["armed_composition"], f["armed_overrides"]
        # 1. critical pressure: shed production
        if "PRESSURE_HIHI" in A:
            return Decision(Action("procedure", name="rate_cut_25"), "unknown", "critical pressure: shed production", source="r1")
        # 2. sustained alarms: arm the standing protections
        sustained = [a for a, age in f["alarm_ages"].items() if age >= 0.25]
        if sustained and not (armed_c and armed_o):
            comps = tuple(c for c, armed in (("safety_overrides", armed_o), ("composition_control", armed_c)) if not armed)
            return Decision(Action("reconfigure", components=comps), "unknown", f"sustained alarms {sustained}", source="r2")
        if "PRESSURE_HI" in A:
            return Decision(_sp("reactor_pressure", max(2500.0, f["P_sp"] - 50.0)), "unknown", "pressure high", source="r3")
        # 3. level alarms: move the corresponding setpoint toward the middle
        for al, key, fld, step in (("RLEVEL_HI", "RL", "reactor_level", -5), ("RLEVEL_LO", "RL", "reactor_level", +5),
                                   ("SLEVEL_HI", "SL", "separator_level", -5), ("SLEVEL_LO", "SL", "separator_level", +5),
                                   ("XLEVEL_HI", "XL", "stripper_level", -5), ("XLEVEL_LO", "XL", "stripper_level", +5)):
            if al in A:
                return Decision(_sp(fld, f[key + "_sp"] + step), "unknown", f"{al}: adjust level setpoint", source="r4")
        if "TEMP_HI" in A or "TEMP_HIHI" in A:
            return Decision(Action("procedure", name="rate_cut_25"), "unknown", "temperature high: shed production", source="r5")
        return Decision(NO_ACTION, "none", "no alarm requires action", source="r0")


class ExpertTable(_Base):
    name = "B2b_expert_table"

    def decide(self, plant, f, trigger, history):
        A = set(f["alarms"])
        armed_c, armed_o = f["armed_composition"], f["armed_overrides"]
        subs = set(f["substituted"])
        # 0. instrument consistency (analytical redundancy) - a lying transmitter blinds its loop
        if "reactor_pressure" not in subs and (abs(f["P_resid"]) > 40.0 or abs(f["dP_rs"]) > 40.0):
            return Decision(Action("substitute_instrument", measurement="reactor_pressure", method="soft"),
                            "sensor_fault", f"pressure transmitter inconsistent (resid {f['P_resid']:+.0f} kPa)", source="e0")
        if "reactor_temperature" not in subs and abs(f["T_resid"]) > 4.0 and not ("TEMP_HI" in A):
            return Decision(Action("substitute_instrument", measurement="reactor_temperature", method="soft"),
                            "sensor_fault", f"temperature transmitter inconsistent (resid {f['T_resid']:+.1f} C)", source="e0b")
        # 1. pressure protection
        if "PRESSURE_HIHI" in A:
            if not armed_o:
                return Decision(Action("reconfigure", components=("safety_overrides",)), "unknown", "arm pressure override", source="e1")
            return Decision(Action("procedure", name="rate_cut_25"), "unknown", "critical pressure: shed production", source="e1b")
        if "PRESSURE_HI" in A:
            # reactant imbalance signature: yA away from nominal -> composition trims
            if not armed_c and abs(f["yA_dev_nom"]) > 1.0:
                return Decision(Action("reconfigure", components=("composition_control",)), "feed_composition_upset",
                                f"pressure high with yA deviation {f['yA_dev_nom']:+.1f}", source="e2")
            if not armed_o:
                return Decision(Action("reconfigure", components=("safety_overrides",)), "unknown", "arm overrides", source="e2b")
            return Decision(Action("procedure", name="rate_cut_25"), "unknown", "pressure high: shed production", source="e2c")
        # 2. temperature high: if cooling valve saturated or not moving, shed load; else lower T sp
        if "TEMP_HI" in A or "TEMP_HIHI" in A:
            if f["mv_r_cooling_water"] >= 99.0 or f["mv_r_cooling_water_std"] < 0.01:
                return Decision(Action("procedure", name="rate_cut_50"), "reactor_cooling_fault", "cooling exhausted/stuck", source="e3")
            return Decision(Action("procedure", name="rate_cut_25"), "reactor_cooling_fault", "temperature high", source="e3b")
        # 3. levels
        for al, key, fld, step in (("RLEVEL_HI", "RL", "reactor_level", -5), ("RLEVEL_LO", "RL", "reactor_level", +5),
                                   ("SLEVEL_HI", "SL", "separator_level", -5), ("SLEVEL_LO", "SL", "separator_level", +5),
                                   ("XLEVEL_HI", "XL", "stripper_level", -5), ("XLEVEL_LO", "XL", "stripper_level", +5)):
            if al in A:
                if not armed_o:
                    return Decision(Action("reconfigure", components=("safety_overrides",)), "unknown", f"{al}: arm overrides", source="e4")
                return Decision(_sp(fld, f[key + "_sp"] + step), "unknown", f"{al}: adjust level setpoint", source="e4b")
        # 4. quality / throughput deviation: composition trims
        if ("GRADE_DEV" in A or abs(f["G_cmd_err"]) > 2.0) and not armed_c:
            return Decision(Action("reconfigure", components=("composition_control",)), "operating_point_change",
                            "product grade off target", source="e5")
        return Decision(NO_ACTION, "none", "steady", source="e9")
