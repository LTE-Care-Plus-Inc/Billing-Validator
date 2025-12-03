import io
from typing import Any
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="HiRasmus Session Checker",
    layout="wide"
)
st.title("Session Checker")
st.markdown("""
This tool reviews **BT 1:1 Direct Service sessions from HiRasmus** and checks whether each session meets
core billing-compliance requirements.

---

### 🔹 Files to Upload

1. **HiRasmus Sessions Export (Excel)**  
   This file contains one row per BT session. It **must include** the following columns:

   - `Status` 
   - `Session` 
   - `Client`
   - `User`
   - `Start time`
   - `End time`  
   - `Duration`
   - `Parent signature time`  
   - `User signature time` 

2. **(Optional) BT Contacts File**  
   A CSV/Excel file containing BT contact information, with columns:

   - `BT Name`  
   - `Phone`  
   - `Email`  

3. **(Optional) External Session List**  
   A CSV/Excel file containing:

   - `Client`
   - `Session Time`

   For each HiRasmus row, if a matching `Session Time` exists for that client
   (by order/sequence), it passes this requirement. If no `Session Time` is found,
   it is flagged.

---

""")

# ---------- Static Config ----------
STATUS_REQUIRED = "Transferred to AlohaABA"
SESSION_REQUIRED = "1:1 BT Direct Service"

MIN_MINUTES = 60    # >= 1 hour
MAX_MINUTES = 360   # <= 6 hours

# Column name for time-adjustment parent approval signature
TIME_ADJ_COL = "Parent’s Signature Approval for Time Adjustment signature"

# Required columns in the uploaded sessions file
REQ_COLS = [
    "Status",
    "Client",
    "Start time",
    "End time",
    "Duration",
    "Session",
    "Parent signature time",
    "User signature time",
    "User",
]


# ---------- Helpers ----------
def read_any(file):
    """Reads CSV or Excel."""
    if file is None:
        return None
    if file.name.lower().endswith((".csv", ".txt")):
        return pd.read_csv(file, dtype=str)
    return pd.read_excel(file, dtype=str, engine="openpyxl")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and string content (strip spaces)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan, "": np.nan})
            )
    return df


def ensure_cols(df: pd.DataFrame, cols, label: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"**{label}** is missing required columns: {missing}")
        return False
    return True


def parse_duration_to_minutes(d: Any) -> float:
    """Convert duration 'HH:MM:SS' (or 'H:MM:SS') → minutes (float)."""
    if pd.isna(d):
        return np.nan
    s = str(d).strip()
    if not s or ":" not in s:
        return np.nan
    try:
        parts = [float(x) for x in s.split(":")]
        if len(parts) == 3:
            h, m, sec = parts
            return h * 60 + m + sec / 60
        elif len(parts) == 2:
            h, m = parts
            return h * 60 + m
    except Exception:
        return np.nan


def export_excel(df: pd.DataFrame) -> bytes:
    """Export a single sheet named 'All'."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="All")
    return buf.getvalue()


def within_time_tol(sig_ts: pd.Timestamp,
                    base_ts: pd.Timestamp,
                    tol_early_min: float = -15,
                    tol_late_min: float = 30) -> bool:
    """
    Compare two timestamps and return True if within allowed window:
      - early: >= tol_early_min  (negative)
      - late:  <= tol_late_min   (positive)
    """
    if pd.isna(sig_ts) or pd.isna(base_ts):
        return False

    diff_min = (sig_ts - base_ts).total_seconds() / 60.0
    return (diff_min >= tol_early_min) and (diff_min <= tol_late_min)


def normalize_name(name: Any) -> str:
    """
    Normalize 'First Middle Last' -> 'Last, First Middle'
    and proper-case each word.
    """
    if pd.isna(name):
        return ""
    s = str(name).strip()
    if not s:
        return ""

    parts = s.split()
    if len(parts) == 1:
        return parts[0].capitalize()

    last_raw = parts[-1]
    first_middle_raw = " ".join(parts[:-1])

    def proper_case_block(block: str) -> str:
        return " ".join(w.capitalize() for w in block.split())

    last = proper_case_block(last_raw)
    first_middle = proper_case_block(first_middle_raw)

    return f"{last}, {first_middle}"


def parse_session_time_range(session_time: Any, base_date: Any):
    """
    Parse '3:00 PM  -  9:00 PM' or '15:00  -  20:00' into (start_dt, end_dt),
    using base_date for the date component.
    """
    if pd.isna(session_time):
        return pd.NaT, pd.NaT

    text = str(session_time).replace("\xa0", " ").strip()
    if not text or "-" not in text:
        return pd.NaT, pd.NaT

    start_str, end_str = [part.strip() for part in text.split("-", 1)]

    # Determine date string
    if pd.isna(base_date):
        date_str = "1900-01-01"
    else:
        date_str = str(base_date)

    start_dt = pd.to_datetime(f"{date_str} {start_str}", errors="coerce")
    end_dt = pd.to_datetime(f"{date_str} {end_str}", errors="coerce")

    return start_dt, end_dt


# ---------- Row-wise check helpers (BASE checks) ----------
def has_time_adjust_sig(row) -> bool:
    """
    True if the second parent signature column exists on this row and is non-empty.
    This does NOT depend on the checkbox – it's just presence.
    """
    if TIME_ADJ_COL not in row.index:
        return False
    val = row.get(TIME_ADJ_COL)
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s not in ("", "nan")


def duration_ok_base(row) -> bool:
    """Base duration check: 60–360 minutes."""
    m = row["Actual Minutes"]
    return (not pd.isna(m)) and (m >= MIN_MINUTES) and (m <= MAX_MINUTES)


def sig_ok_base(row) -> bool:
    """
    Base signature check:
    - Uses asymmetric tolerance (-15 early, +30 late).
    - If no signatures at all, pass.
    """
    base_ts = row.get("_End_dt", pd.NaT)
    if pd.isna(base_ts):
        return False

    parent_sig_ts = row.get("_ParentSig_dt", pd.NaT)
    user_sig_ts = row.get("_UserSig_dt", pd.NaT)

    checks = []

    if not pd.isna(parent_sig_ts):
        checks.append(within_time_tol(parent_sig_ts, base_ts))

    if not pd.isna(user_sig_ts):
        checks.append(within_time_tol(user_sig_ts, base_ts))

    # No signatures at all → pass
    if pd.isna(parent_sig_ts) and pd.isna(user_sig_ts):
        return True

    # All present signatures must pass tolerance
    return all(checks)


def external_match_ok(row) -> bool:
    """
    External session requirement:
    - If there is no 'Has External Session' column (no 2nd file), ignore this rule.
    - If it exists, row only passes if Has External Session is True.
    """
    if "Has External Session" not in row.index:
        return True
    return bool(row["Has External Session"])


# ---------- Combined evaluation (uses checkbox + override) ----------
def evaluate_row(row) -> dict:
    """
    Returns dict of booleans:
      - duration_ok
      - sig_ok
      - ext_ok
      - has_time_adj_sig
      - overall_pass
    Applies override logic when checkbox is ON:
      - If USE_TIME_ADJ_OVERRIDE and row has second parent signature:
          signature issues are ignored (sig_ok = True)
      - Duration is ALWAYS enforced.
    """
    dur_base = duration_ok_base(row)
    sig_base = sig_ok_base(row)
    ext_ok = external_match_ok(row)
    adj_sig = has_time_adjust_sig(row)

    # Start from base
    duration_ok = dur_base
    sig_ok = sig_base

    # If checkbox is ON and second parent signature exists, ignore signature issues
    if USE_TIME_ADJ_OVERRIDE and adj_sig:
        sig_ok = True

    overall = duration_ok and sig_ok and ext_ok

    return {
        "duration_ok": duration_ok,
        "sig_ok": sig_ok,
        "ext_ok": ext_ok,
        "has_time_adj_sig": adj_sig,
        "overall_pass": overall,
        "duration_ok_base": dur_base,
        "sig_ok_base": sig_base,
    }


def get_failure_reasons(row) -> str:
    eval_res = evaluate_row(row)
    reasons = []

    # Duration failures (never overridden)
    if not eval_res["duration_ok"]:
        actual_min = row.get("Actual Minutes")
        if pd.isna(actual_min):
            reasons.append("Missing Duration data")
        else:
            reasons.append(
                f"Duration ({actual_min:.0f} min) is outside allowed range "
                f"({MIN_MINUTES}-{MAX_MINUTES} min)"
            )

    # Signature failures (after considering override)
    if not eval_res["sig_ok"]:
        if USE_TIME_ADJ_OVERRIDE and not eval_res["has_time_adj_sig"]:
            reasons.append(
                "Signature not within -15/+30 minutes of end time (no time-adjustment signature override)"
            )
        else:
            reasons.append("Signature not within -15/+30 minutes of end time")

    # External session failures
    if not eval_res["ext_ok"]:
        reasons.append("No matching Session Time in external file")

    return "; ".join(reasons) if reasons else "PASS"


# ---------- Sidebar: BT Contacts (optional) ----------
st.sidebar.header("BT Contacts (optional)")
bt_contacts_file = st.sidebar.file_uploader(
    "Upload BT Contacts (BT Name, Phone, Email)",
    type=["csv", "xlsx"],
    key="bt_contacts",
)

# ---------- Main UI ----------
st.subheader("1) Upload Sessions File")
sessions_file = st.file_uploader(
    "Sessions File — HiRasmus Export (Excel)",
    type=["xlsx", "xls"],
)

if not sessions_file:
    st.info("Upload the sessions Excel file to continue.")
    st.stop()

st.subheader("1b) Optional: Upload External Session List")
external_sessions_file = st.file_uploader(
    "External Session List (Client, Session Time)",
    type=["csv", "xlsx"],
    key="external_sessions",
)

# Checkbox: use second parent signature as override for signature timing only
USE_TIME_ADJ_OVERRIDE = st.checkbox(
    f"Use '{TIME_ADJ_COL}' as a signature override (only for sessions that failed signature timing; duration still enforced)",
    value=False,
)

# ---------- Read and normalize Sessions file ----------
df = pd.read_excel(sessions_file, dtype=object)
df = normalize_cols(df)

if not ensure_cols(df, REQ_COLS, "Sessions File"):
    st.stop()

# If override is enabled, ensure the column exists (otherwise user probably mis-clicked)
if USE_TIME_ADJ_OVERRIDE:
    if not ensure_cols(df, [TIME_ADJ_COL], "Sessions File"):
        st.stop()

# ---------- Prefilter by Status + Session ----------
df_f = df[
    (df["Status"].astype(str).str.strip() == STATUS_REQUIRED)
    & (df["Session"].astype(str).str.strip() == SESSION_REQUIRED)
].copy()

st.caption(
    f'Prefilters → Status == "{STATUS_REQUIRED}" | Session == "{SESSION_REQUIRED}"'
)
st.write({"Rows (before/after)": f"{len(df)}/{len(df_f)}"})

# ---------- Parse Duration (initially from HiRasmus Duration) ----------
df_f["Actual Minutes"] = df_f["Duration"].apply(parse_duration_to_minutes)

# ---------- Parse Time Columns from HiRasmus (for date + backup) ----------
start_dt = pd.to_datetime(df_f["Start time"], errors="coerce")
end_dt = pd.to_datetime(df_f["End time"], errors="coerce")

df_f["_Start_dt"] = start_dt
df_f["_End_dt"] = end_dt  # may be overridden by external Session Time
df_f["_ParentSig_dt"] = pd.to_datetime(df_f["Parent signature time"], errors="coerce")
df_f["_UserSig_dt"] = pd.to_datetime(df_f["User signature time"], errors="coerce")

# ---------- Date column (prefer End time date, fall back to Start time date) ----------
date_from_end = df_f["_End_dt"].dt.date
date_from_start = df_f["_Start_dt"].dt.date

df_f["Date"] = date_from_end
df_f.loc[df_f["Date"].isna(), "Date"] = date_from_start[df_f["Date"].isna()]

# ---------- Normalize names ----------
df_f["Staff Name"] = df_f["User"].apply(normalize_name)
df_f["Client Name"] = df_f["Client"].apply(normalize_name)

# ---------- Optional: Match to external Client/Session Time list ----------
# ---------- Optional: Match to external Client/Session Time list ----------
if external_sessions_file is not None:
    ext_df = read_any(external_sessions_file)
    if ext_df is not None:
        ext_df = normalize_cols(ext_df)
        if ensure_cols(ext_df, ["Client", "Session Time"], "External Sessions File"):
            # Normalize client names in the external file the same way
            ext_df["Client Name"] = ext_df["Client"].apply(normalize_name)

            # For each client, assign an order index (0,1,2,...) in the external file
            # IMPORTANT: no sorting here, we keep the original order of ext_df
            ext_df["SessionIndex"] = ext_df.groupby("Client Name").cumcount()

            # In the HiRasmus dataframe, assign per-client order index
            # IMPORTANT: no sorting here, we keep the original order of df_f
            df_f["SessionIndex"] = df_f.groupby("Client Name").cumcount()

            # Merge on (Client Name, SessionIndex) WITHOUT changing df_f order
            df_f = df_f.merge(
                ext_df[["Client Name", "SessionIndex", "Session Time"]],
                on=["Client Name", "SessionIndex"],
                how="left",
                sort=False,
            )

            # True if this HiRasmus row has a matching external session
            df_f["Has External Session"] = df_f["Session Time"].notna()

            # ---- Use Session Time to override start/end + duration when available ----
            df_f[["_ExtStart_dt", "_ExtEnd_dt"]] = df_f.apply(
                lambda r: pd.Series(
                    parse_session_time_range(r.get("Session Time"), r.get("Date"))
                ),
                axis=1,
            )

            has_ext_valid = (
                df_f["Has External Session"]
                & df_f["_ExtStart_dt"].notna()
                & df_f["_ExtEnd_dt"].notna()
            )

            # Override Actual Minutes and _End_dt for rows with valid external times
            df_f.loc[has_ext_valid, "Actual Minutes"] = (
                (df_f.loc[has_ext_valid, "_ExtEnd_dt"] - df_f.loc[has_ext_valid, "_ExtStart_dt"])
                .dt.total_seconds()
                / 60.0
            )
            df_f.loc[has_ext_valid, "_End_dt"] = df_f.loc[has_ext_valid, "_ExtEnd_dt"]
        else:
            st.warning(
                "External sessions file uploaded, but required columns are missing."
            )


# ---------- Attach BT contacts via fuzzy match ----------
df_f["Phone"] = ""
df_f["Email"] = ""

if bt_contacts_file is not None:
    bt_df = read_any(bt_contacts_file)
    if bt_df is not None:
        bt_df = normalize_cols(bt_df)

        bt_required = {"BT Name", "Phone", "Email"}
        bt_missing = bt_required - set(bt_df.columns)
        if bt_missing:
            st.error(f"BT Contacts file is missing: {sorted(bt_missing)}")
        else:
            # Normalize BT Name in same way as Staff Name
            bt_df["BT_formatted"] = bt_df["BT Name"].apply(normalize_name)

            def norm_name(s: str) -> str:
                s = str(s).strip().lower()
                s = s.replace(",", " ")
                s = " ".join(s.split())
                return s

            bt_df["bt_norm"] = bt_df["BT_formatted"].apply(norm_name)

            staff_to_phone = {}
            staff_to_email = {}

            staff_unique = df_f["Staff Name"].dropna().unique()

            for staff in staff_unique:
                staff_norm = norm_name(staff)
                best_score = 0.0
                best_row = None

                for _, bt_row in bt_df.iterrows():
                    bt_name_norm = bt_row["bt_norm"]
                    score = SequenceMatcher(None, staff_norm, bt_name_norm).ratio()
                    if score > best_score:
                        best_score = score
                        best_row = bt_row

                if best_row is not None and best_score >= 0.8:
                    staff_to_phone[staff] = best_row["Phone"]
                    staff_to_email[staff] = best_row["Email"]

            df_f["Phone"] = df_f["Staff Name"].map(staff_to_phone)
            df_f["Email"] = df_f["Staff Name"].map(staff_to_email)

# ---------- Run Checks ----------
eval_results = df_f.apply(evaluate_row, axis=1, result_type="expand")
# result_type="expand" gives a DataFrame if evaluate_row returns dict, but older Streamlit/Pandas
# might behave differently, so we handle both possibilities:

if isinstance(eval_results, pd.DataFrame):
    df_f["_DurationOk"] = eval_results["duration_ok"]
    df_f["_SigOk"] = eval_results["sig_ok"]
    df_f["_ExtOk"] = eval_results["ext_ok"]
    df_f["_HasTimeAdjSig"] = eval_results["has_time_adj_sig"]
    df_f["_OverallPass"] = eval_results["overall_pass"]
else:
    # Fallback: eval_results is a Series of dicts
    df_f["_DurationOk"] = eval_results.apply(lambda r: r["duration_ok"])
    df_f["_SigOk"] = eval_results.apply(lambda r: r["sig_ok"])
    df_f["_ExtOk"] = eval_results.apply(lambda r: r["ext_ok"])
    df_f["_HasTimeAdjSig"] = eval_results.apply(lambda r: r["has_time_adj_sig"])
    df_f["_OverallPass"] = eval_results.apply(lambda r: r["overall_pass"])

df_f["Failure Reasons"] = df_f.apply(get_failure_reasons, axis=1)

# ---------- Pretty-print time columns ----------
for col in [
    "Start time",
    "End time",
    "Parent signature time",
    "User signature time",
]:
    if col in df_f.columns:
        df_f[col] = (
            pd.to_datetime(df_f[col], errors="coerce")
            .dt.strftime("%I:%M:%S %p")
            .fillna("")
        )

# ---------- Display ----------
display_cols = [
    "Status",
    "Session",
    "Date",
    "Client Name",
    "Staff Name",
    "Phone",
    "Email",
    "Start time",
    "End time",
    "Duration",
    "Actual Minutes",
    "Parent signature time",
    "User signature time",
    "Session Time",           # from external file (if provided)
    "Has External Session",   # True/False (if external file provided)
]

# Show the time-adjustment approval column if present
if TIME_ADJ_COL in df_f.columns:
    display_cols.append(TIME_ADJ_COL)

display_cols.extend([
    "_DurationOk",
    "_SigOk",
    "_ExtOk",
    "_HasTimeAdjSig",
    "Failure Reasons",
])

present_cols = [c for c in display_cols if c in df_f.columns]

st.subheader("2) Results")
st.dataframe(df_f[present_cols], use_container_width=True, height=560)

st.caption("Summary")
summary = pd.DataFrame(
    {
        "Total (after filters)": [len(df_f)],
        "Pass": [int(df_f["_OverallPass"].sum())],
        "Fail": [int((~df_f["_OverallPass"]).sum())],
    }
)
st.table(summary)

# ---------- Export ----------
xlsx_all = export_excel(df_f[present_cols])
xlsx_clean = export_excel(df_f[df_f["_OverallPass"]][present_cols])
xlsx_flagged = export_excel(df_f[~df_f["_OverallPass"]][present_cols])

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "⬇️ Download All", data=xlsx_all, file_name="all_sessions.xlsx"
    )
with c2:
    st.download_button(
        "✅ Passed Only", data=xlsx_clean, file_name="clean_sessions.xlsx"
    )
with c3:
    st.download_button(
        "⚠️ Failed Only", data=xlsx_flagged, file_name="flagged_sessions.xlsx"
    )
