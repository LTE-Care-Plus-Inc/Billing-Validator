# ===========================
# validators.py
# ===========================
import numpy as np
import pandas as pd

from constants import (
    TIME_ADJ_COL,
    MIN_MINUTES,
    MAX_MINUTES,
    DAILY_MAX_MINUTES,
    MIN_SESSION_GAP_MINUTES,
)
from utils import within_time_tol

# These are set by app.py at runtime via configure()
BILLING_TOL = 8
DAILY_TOL = 8
SIG_TOL_EARLY = -8
USE_TIME_ADJ_OVERRIDE = True


def configure(billing_tol: int, daily_tol: int, sig_tol_early: int, use_time_adj_override: bool):
    """Called by app.py after sidebar widgets are rendered to inject runtime settings."""
    global BILLING_TOL, DAILY_TOL, SIG_TOL_EARLY, USE_TIME_ADJ_OVERRIDE
    BILLING_TOL = billing_tol
    DAILY_TOL = daily_tol
    SIG_TOL_EARLY = sig_tol_early
    USE_TIME_ADJ_OVERRIDE = use_time_adj_override


# =========================================================
# Row-level checks
# =========================================================
def has_time_adjust_sig(row) -> bool:
    if TIME_ADJ_COL not in row.index:
        return False
    val = row.get(TIME_ADJ_COL)
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s not in ("", "nan")


def duration_ok_base(row) -> bool:
    m = row.get("Actual Minutes")
    if pd.isna(m):
        return False
    if m < MIN_MINUTES:
        return False
    if m <= MAX_MINUTES:
        return True
    return (m - MAX_MINUTES) < BILLING_TOL


def external_time_ok(row) -> bool:
    if "Has External Session" not in row.index:
        return True
    if not bool(row.get("Has External Session")):
        return False
    return (not pd.isna(row.get("_ExtStart_dt"))) and (not pd.isna(row.get("_ExtEnd_dt")))


def note_attendance_ok(row) -> bool:
    client_present = row.get("_Note_ClientPresent")
    bt_present = row.get("_Note_BTPresent")
    parent_present = row.get("_Note_ParentPresent")
    sibling_present = row.get("_Note_SiblingPresent")

    if all(pd.isna(x) for x in [client_present, bt_present, parent_present, sibling_present]):
        return True

    if not bool(client_present):
        return False
    if not bool(bt_present):
        return False
    if not (bool(parent_present) or bool(sibling_present)):
        return False
    return True


def note_parse_ok(row) -> bool:
    if "Note Parse PASS" not in row.index:
        return True
    val = row.get("Note Parse PASS")
    if pd.isna(val):
        return False
    return bool(val)


def sig_ok_base(row) -> bool:
    base_ts = row.get("_End_dt", pd.NaT)
    parent_sig_ts = row.get("_ParentSig_dt", pd.NaT)

    if pd.isna(parent_sig_ts):
        return False
    if pd.isna(base_ts):
        return False

    if USE_TIME_ADJ_OVERRIDE and has_time_adjust_sig(row):
        return True

    return within_time_tol(parent_sig_ts, base_ts, SIG_TOL_EARLY)


def daily_total_ok(row) -> bool:
    m = row.get("Daily Minutes")
    if pd.isna(m):
        return True
    return m < (DAILY_MAX_MINUTES + DAILY_TOL)


def evaluate_row(row) -> dict:
    dur_base = duration_ok_base(row)
    sig_base = sig_ok_base(row)
    ext_ok = external_time_ok(row)
    adj_sig = has_time_adjust_sig(row)
    daily_ok_val = daily_total_ok(row)
    note_ok_val = note_attendance_ok(row)
    note_parse_ok_val = note_parse_ok(row)

    # Gap check disabled
    gap_ok_val = row.get("_SessionGapOk", True)
    if pd.isna(gap_ok_val):
        gap_ok_val = True
    gap_ok_val = bool(gap_ok_val)

    duration_ok = dur_base
    sig_ok = sig_base

    overall = (
        duration_ok
        and sig_ok
        and ext_ok
        and daily_ok_val
        and note_ok_val
        and note_parse_ok_val
        and gap_ok_val
    )

    return {
        "duration_ok": duration_ok,
        "sig_ok": sig_ok,
        "ext_ok": ext_ok,
        "daily_ok": daily_ok_val,
        "note_ok": note_ok_val,
        "note_parse_ok": note_parse_ok_val,
        "gap_ok": gap_ok_val,
        "has_time_adj_sig": adj_sig,
        "overall_pass": overall,
        "duration_ok_base": dur_base,
        "sig_ok_base": sig_base,
    }


def get_failure_reasons(row) -> str:
    eval_res = evaluate_row(row)
    reasons = []

    if not eval_res.get("note_parse_ok", True):
        note_pass = row.get("Note Parse PASS", np.nan)
        note_errs = row.get("Note Compliance Errors", np.nan)

        if pd.isna(note_pass):
            reasons.append("No matching session note found in PDF (required)")
        else:
            if pd.isna(note_errs) or str(note_errs).strip() == "":
                reasons.append("Session note failed compliance checks")
            else:
                reasons.append(f"Session note failed compliance checks: {note_errs}")

    if not eval_res["ext_ok"]:
        stime = row.get("Session Time", "")
        if pd.isna(stime) or str(stime).strip() == "":
            reasons.append("No Session time on note")
        else:
            reasons.append("Session Time invalid: must be same-day and end before 10:00 PM")

    if not eval_res["duration_ok"]:
        actual_min = row.get("Actual Minutes")
        if pd.isna(actual_min):
            reasons.append("Missing Duration data")
        else:
            reasons.append(
                f"Duration ({actual_min:.0f} min) must be between {MIN_MINUTES} and {MAX_MINUTES} minutes"
            )

    if not eval_res["sig_ok"]:
        if pd.isna(row.get("_ParentSig_dt", pd.NaT)):
            reasons.append("Missing Adult Caregiver signature time (required)")
        else:
            reasons.append("Parent signature too early.")

    if not eval_res.get("daily_ok", True):
        daily_min = row.get("Daily Minutes")
        if not pd.isna(daily_min):
            reasons.append(
                f"Total daily duration for this BT on {row.get('Date')} "
                f"({daily_min:.0f} min) exceeded"
            )

    if not eval_res.get("gap_ok", True):
        gap_min = row.get("_SessionGapMinutes")
        if pd.isna(gap_min) or gap_min < 0:
            reasons.append("Same-day sessions overlap for this client")
        else:
            reasons.append(
                f"Same-day sessions for this client are only {gap_min:.0f} min apart "
                f"(minimum {MIN_SESSION_GAP_MINUTES} min required)"
            )
            
    if not eval_res.get("note_ok", True):
        missing = []

        client_ok = bool(row.get("_Note_ClientPresent"))
        bt_ok = bool(row.get("_Note_BTPresent"))
        parent_ok = bool(row.get("_Note_ParentPresent"))
        sibling_ok = bool(row.get("_Note_SiblingPresent"))

        if not client_ok:
            missing.append("Client")
        if not bt_ok:
            missing.append("BT/RBT")
        if not (parent_ok or sibling_ok):
            missing.append("Parent/Caregiver or Sibling")

        reasons.append("Session note attendance issue: missing " + ", ".join(missing))

    return "; ".join(reasons) if reasons else "PASS"