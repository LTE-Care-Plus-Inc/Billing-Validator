import io
import re
from typing import Any
from difflib import SequenceMatcher

import fitz  
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="HiRasmus Billing Session Checker",
    layout="wide",
)
st.title("HiRasmus Billing Session Checker")

# -----------------------------
# GLOBAL CONFIG
# -----------------------------
STATUS_REQUIRED = "Transferred to AlohaABA"
SESSION_REQUIRED = "1:1 BT Direct Service"

MIN_MINUTES = 60    # >= 1 hour
MAX_MINUTES = 360   # <= 6 hours
BILLING_TOL_DEFAULT = 8      # up to 8 min over MAX allowed
DAILY_MAX_MINUTES = 480      # <= 8 hours per BT per day

# Column name for time-adjustment parent approval signature
TIME_ADJ_COL = "Parent’s Signature Approval for Time Adjustment signature"

# Required columns in the uploaded sessions file
REQ_COLS = [
    "Status",
    "AlohaABA Appointment ID",
    "Client",
    "Duration",
    "Session",
    "Parent signature time",
    "User signature time",
    "User",
    "Start date",   # mandatory date column from HiRasmus
]

# -----------------------------
# PDF → TEXT (TOOLS TAB)
# -----------------------------
def pdf_bytes_to_text(pdf_bytes: bytes, preserve_layout: bool = True) -> str:
    """
    Convert PDF bytes to a single text string.
    Adds '--- Page X ---' separators between pages.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    all_text = []

    for page_num in range(total_pages):
        page = doc[page_num]
        if preserve_layout:
            text = page.get_text(
                "text",
                flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES,
            )
        else:
            text = page.get_text("text")

        all_text.append(f"--- Page {page_num + 1} ---\n{text}\n\n")

    doc.close()
    return "".join(all_text)


def parse_notes(text: str):
    """
    Parse client notes from text extracted from PDF.
    Returns a list of dicts with at least:
      - Client
      - Rendering Provider
      - Date
      - Session Time
    """
    # Split notes by "Client:" (but keep the keyword)
    blocks = re.split(r"(?=Client:)", text)

    results = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # ----------------------------
        # Client Name
        # ----------------------------
        client_match = re.search(r"Client:\s*([^\n,]+)", block)
        client_name = client_match.group(1).strip() if client_match else ""

        if not client_name:
            continue

        # ----------------------------
        # Rendering Provider
        # ----------------------------
        provider_match = re.search(r"Rendering Provider:\s*([^\n]+)", block)
        provider = provider_match.group(1).strip() if provider_match else ""

        # ----------------------------
        # Date (YYYY/MM/DD)
        # ----------------------------
        date_match = re.search(r"Date:\s*([0-9]{4}/[0-9]{2}/[0-9]{2})", block)
        date_value = date_match.group(1) if date_match else ""

        # ----------------------------
        # Session Time
        # - if just "-" or blank → ""
        # - if real time range → keep it
        # ----------------------------
        session_time_match = re.search(
            r"Session Time:\s*(?:"
            r"([0-9]{1,2}:[0-9]{2}\s*(?:AM|PM)?\s*-\s*[0-9]{1,2}:[0-9]{2}\s*(?:AM|PM)?)"
            r"|-"  # OR literal dash
            r")?",
            block,
            re.IGNORECASE,
        )

        if session_time_match:
            session_time = session_time_match.group(1)
            if session_time is None:
                session_time = ""
        else:
            session_time = ""

        results.append(
            {
                "Client": client_name,
                "Rendering Provider": provider,
                "Date": date_value,
                "Session Time": session_time,
            }
        )

    return results


def notes_to_excel_bytes(results, sheet_name="Notes") -> bytes:
    """
    Convert parsed notes list[dict] → Excel file bytes.
    """
    df = pd.DataFrame(results)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output


# -----------------------------
# SHARED HELPERS
# -----------------------------
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


def within_time_tol(
    sig_ts: pd.Timestamp,
    base_ts: pd.Timestamp,
    tol_early_min: float,
    tol_late_min: float,
) -> bool:
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


# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("Settings")

with st.sidebar.expander("Signature Settings", expanded=False):
    SIG_TOL_EARLY = st.number_input(
        "Signature early tolerance (minutes, negative)",
        value=-15,
        step=1,
        help="Earliest allowed time before session end (e.g., -15 means 15 minutes before).",
    )
    SIG_TOL_LATE = st.number_input(
        "Signature late tolerance (minutes, positive)",
        value=480,
        min_value=0,
        step=1,
        help="Latest allowed time after session end (e.g., 30 means up to 30 minutes after).",
    )

with st.sidebar.expander("Duration / Billing Settings", expanded=False):
    BILLING_TOL = st.number_input(
        "Billing duration tolerance (minutes)",
        value=BILLING_TOL_DEFAULT,
        min_value=0,
        step=1,
        help=(
            "If a session's duration is outside the base range "
            f"({MIN_MINUTES}-{MAX_MINUTES} min) by at most this many minutes, "
            "it will still be treated as OK."
        ),
    )

# BT Contacts (optional) – used in Session Checker tab
st.sidebar.header("BT Contacts (optional)")
bt_contacts_file = st.sidebar.file_uploader(
    "Upload BT Contacts (BT Name, Phone, Email)",
    type=["csv", "xlsx"],
    key="bt_contacts",
)

# Instructions toggle
with st.expander("Instructions", expanded=False):
    st.markdown(
        """
This tool reviews **BT 1:1 Direct Service sessions from HiRasmus** and checks whether each session meets
core billing-compliance requirements.

There are **two tabs** in this app:

- **Tools – Extract External Sessions**: Upload the *Session Notes PDF* from HiRasmus.  
  The app parses it and builds an **External Session List** (`Client`, `Session Time`) and stores it for use in the Session Checker.  
  Rows with blank `Session Time` are **kept** (they just won’t count as a matched external session).

- **Session Checker**: Upload the HiRasmus sessions Excel export and run all compliance checks.  
  The checker automatically uses the External Session List from the Tools tab, unless you choose to override it
  by uploading another external file.

---

### 🔹 Files to Upload (Session Checker tab)

1. **HiRasmus Sessions Export (Excel)**  
   This file contains one row per BT session. It **must include** the following columns:

   - `Status`  
   - `Session`
   - `Aloha Appointment ID`  
   - `Client`  
   - `User`  
   - `Start date`  
   - `Duration`  
   - `Parent signature time`  
   - `User signature time`  

2. **(Optional) BT Contacts File**  
   Uploaded via the **sidebar** as a CSV/Excel file with:

   - `BT Name`  
   - `Phone`  
   - `Email`  

   The app will fuzzy-match `BT Name` to the HiRasmus `User`/`Staff Name` and attach phone/email.

3. **External Session List**  
   By default this comes from the **Tools tab** when you upload the Session Notes PDF.  
   You can also manually upload a file here to override it.

   Required columns:

   - `Client`  
   - `Session Time`  

   Notes:
   - All rows with a valid `Client` are kept, even if `Session Time` is blank.  
   - For each HiRasmus row, the app matches on **Client (normalized)** and **sequence/order** (first note, second note, etc.).  
   - If a matched row has a **non-blank `Session Time`**, it is used to:
     - mark `Has External Session = True`, and  
     - override the session **start/end time** and **duration**.
   - If the matched external row has a **blank `Session Time`**, it will **not** count as a valid external session for checks
     and will behave like “no session time found”.

---

### 🔹 Signature Timing & Overrides

- Signature timing tolerance is controlled in the **sidebar**:
  - **Early tolerance** (negative minutes, e.g. `-15`)
  - **Late tolerance** (positive minutes, e.g. `30`)

- You can enable an option to use  
  **“Parent’s Signature Approval for Time Adjustment signature”**  
  as an override:
  - If enabled and that field is present on a row, **signature timing failures are ignored** for that row.
  - **Duration rules and daily 8-hour limit are always enforced**.

After all checks, the app:

- Shows a detailed table with pass/fail indicators and reasons  
- Excludes test client **“Marry Wang Demo”** from the final Excel exports  
- Provides three downloads: **All**, **Passed Only**, and **Failed Only** sessions in Tab 2,  
  and four billing files in Tab 3.
        """
    )

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "1️⃣ Tools – Extract External Sessions",
    "2️⃣ Session Checker",
    "3️⃣ Billed Checker",
])

# Keep external sessions DF across tabs
if "external_sessions_df" not in st.session_state:
    st.session_state["external_sessions_df"] = None

# =========================================================
# TAB 1: TOOLS – PDF → EXTERNAL SESSION LIST
# =========================================================
with tab1:
    st.header("Session Notes PDF → External Session List")

    pdf_file = st.file_uploader(
        "Upload HiRasmus Session Notes PDF",
        type=["pdf"],
        key="notes_pdf",
    )

    if pdf_file is not None:
        try:
            # Extract text (auto-run)
            text = pdf_bytes_to_text(pdf_file.read(), preserve_layout=True)

            # Parse notes
            results = parse_notes(text)
            if not results:
                st.warning("No notes found in the PDF. Make sure it contains 'Client:' blocks.")
            else:
                notes_df = pd.DataFrame(results)
                st.success(f"Parsed {len(notes_df)} notes from PDF.")

                # Show notes preview in a toggle
                with st.expander("Preview parsed notes", expanded=False):
                    st.dataframe(notes_df.head(50))

                # Save external session list (Client, Session Time) into session_state
                ext_df = notes_df[["Client", "Session Time"]].copy()
                ext_df = normalize_cols(ext_df)

                # Keep rows where Client is present, but DO NOT filter out blank Session Time
                ext_df = ext_df[ext_df["Client"].notna()]

                st.session_state["external_sessions_df"] = ext_df

                st.info(
                    f"Generated {len(ext_df)} external session rows. "
                    "These will be used automatically in the **Session Checker** tab."
                )

                # Download external sessions Excel
                ext_xlsx = notes_to_excel_bytes(ext_df.to_dict(orient="records"), sheet_name="External Sessions")
                st.download_button(
                    "⬇️ Download External Session List (.xlsx)",
                    data=ext_xlsx,
                    file_name="external_sessions_from_pdf.xlsx",
                )

        except Exception as e:
            st.error(f"Error during PDF processing: {e}")

# =========================================================
# TAB 2: SESSION CHECKER
# =========================================================
# Row-wise check helpers depend on sidebar settings and global flags
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
    """Base duration check with over-max billing tolerance.

    - Always fail if duration < MIN_MINUTES.
    - Pass if MIN_MINUTES <= duration <= MAX_MINUTES.
    - If duration > MAX_MINUTES, allow up to BILLING_TOL minutes over the max.
      Only flag if it exceeds MAX_MINUTES + BILLING_TOL.
    """
    m = row["Actual Minutes"]
    if pd.isna(m):
        return False

    # Too short -> always fail (no tolerance below min)
    if m < MIN_MINUTES:
        return False

    # Within base range -> OK
    if m <= MAX_MINUTES:
        return True

    # Over max -> allow up to BILLING_TOL minutes over
    over_by = m - MAX_MINUTES
    if over_by <= BILLING_TOL:
        return True

    # More than BILLING_TOL minutes over -> fail
    return False


def external_match_ok(row) -> bool:
    """
    External session requirement:
    - If there is no 'Has External Session' column (no 2nd file), ignore this rule.
    - If it exists, row only passes if Has External Session is True.
    """
    if "Has External Session" not in row.index:
        return True
    return bool(row["Has External Session"])


def sig_ok_base(row) -> bool:
    """
    Signature timing check against session end time.

    Uses:
    - _End_dt as the base timestamp (from internal PDF-derived times)
    - _ParentSig_dt and _UserSig_dt for signatures
    - SIG_TOL_EARLY / SIG_TOL_LATE from the sidebar
    """
    base_ts = row.get("_End_dt", pd.NaT)

    parent_sig_ts = row.get("_ParentSig_dt", pd.NaT)
    user_sig_ts = row.get("_UserSig_dt", pd.NaT)

    # If both signatures are missing, treat as OK (no timing to check)
    if pd.isna(parent_sig_ts) and pd.isna(user_sig_ts):
        return True

    if pd.isna(base_ts):
        # There ARE signatures but no internal end time to compare against
        return False

    checks = []

    if not pd.isna(parent_sig_ts):
        checks.append(
            within_time_tol(
                parent_sig_ts,
                base_ts,
                SIG_TOL_EARLY,
                SIG_TOL_LATE,
            )
        )

    if not pd.isna(user_sig_ts):
        checks.append(
            within_time_tol(
                user_sig_ts,
                base_ts,
                SIG_TOL_EARLY,
                SIG_TOL_LATE,
            )
        )

    # require all present signatures to be within tolerance
    return all(checks) if checks else False


def daily_total_ok(row) -> bool:
    """
    Check if total minutes for this Staff on this Date exceed DAILY_MAX_MINUTES.
    Uses precomputed 'Daily Minutes'.
    """
    m = row.get("Daily Minutes")
    if pd.isna(m):
        return True
    return m <= DAILY_MAX_MINUTES


# Combined evaluation (uses checkbox + override)
def evaluate_row(row) -> dict:
    """
    Returns dict of booleans:
      - duration_ok
      - sig_ok
      - ext_ok
      - daily_ok
      - has_time_adj_sig
      - overall_pass
    Applies override logic when checkbox is ON:
      - If USE_TIME_ADJ_OVERRIDE and row has second parent signature:
          signature issues are ignored (sig_ok = True)
      - Duration and daily limits are ALWAYS enforced.
    """
    dur_base = duration_ok_base(row)
    sig_base = sig_ok_base(row)
    ext_ok = external_match_ok(row)
    adj_sig = has_time_adjust_sig(row)
    daily_ok_val = daily_total_ok(row)

    # Start from base
    duration_ok = dur_base
    sig_ok = sig_base

    # If checkbox is ON and second parent signature exists, ignore signature issues
    if USE_TIME_ADJ_OVERRIDE and adj_sig:
        sig_ok = True

    overall = duration_ok and sig_ok and ext_ok and daily_ok_val

    return {
        "duration_ok": duration_ok,
        "sig_ok": sig_ok,
        "ext_ok": ext_ok,
        "daily_ok": daily_ok_val,
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
                f"({MIN_MINUTES}-{MAX_MINUTES} min, +{BILLING_TOL} min tolerance over max)"
            )

    # Signature failures (after considering override)
    if not eval_res["sig_ok"]:
        tol_str = f"{SIG_TOL_EARLY}/+{SIG_TOL_LATE}"
        if USE_TIME_ADJ_OVERRIDE and not eval_res["has_time_adj_sig"]:
            reasons.append(
                f"Signature not within {tol_str} minutes of end time "
                "(no time-adjustment signature override)"
            )
        else:
            reasons.append(f"Signature not within {tol_str} minutes of end time")

    # External session failures
    if not eval_res["ext_ok"]:
        reasons.append("Session time empty on note")

    # Daily total failures
    if not eval_res.get("daily_ok", True):
        daily_min = row.get("Daily Minutes")
        if not pd.isna(daily_min):
            reasons.append(
                f"Total daily duration for this BT on {row.get('Date')} "
                f"({daily_min:.0f} min) exceeds {DAILY_MAX_MINUTES} min"
            )

    return "; ".join(reasons) if reasons else "PASS"


with tab2:
    st.header("Session Checker")

    # ---------- Main UI ----------
    st.subheader("Upload HiRamsus Excel File")
    sessions_file = st.file_uploader(
        "Upload HiRamsus Excel File",
        type=["xlsx", "xls"],
        key="sessions_file",
    )

    if not sessions_file:
        st.info("Upload the HiRamsus Excel file to continue.")
        st.stop()

    external_sessions_df = st.session_state.get("external_sessions_df", None)

    if external_sessions_df is not None and len(external_sessions_df) > 0:
        st.success(
            f"Using {len(external_sessions_df)} external session rows from the **Tools** tab."
        )
    else:
        st.warning(
            "No external sessions found from the Tools tab yet. "
            "You can upload an external session file manually."
        )
        manual_external_file = st.file_uploader(
            "External Session List (Client, Session Time)",
            type=["csv", "xlsx"],
            key="external_sessions_manual_only",
        )
        if manual_external_file is not None:
            external_sessions_df = read_any(manual_external_file)

    # Checkbox: use second parent signature as override for signature timing only
    global USE_TIME_ADJ_OVERRIDE
    USE_TIME_ADJ_OVERRIDE = st.toggle(
        f"Use '{TIME_ADJ_COL}' as a signature override (only for sessions that failed signature timing; duration & daily limit still enforced)",
        value=False,
    )

    # ---------- Read and normalize Sessions file ----------
    df = pd.read_excel(sessions_file, dtype=object)
    df = normalize_cols(df)

    # Handle possible "Start Date" vs "Start date" header
    if "Start date" not in df.columns and "Start Date" in df.columns:
        df = df.rename(columns={"Start Date": "Start date"})

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

   # ---------- Parse Duration (initially from HiRasmus Duration) ----------
    df_f["Actual Minutes"] = df_f["Duration"].apply(parse_duration_to_minutes)

    # ---------- Parse Date from Start date (keep only text before first space) ----------
    start_raw = df_f["Start date"].astype(str).str.strip()
    # e.g. "11/30/2025  5:00:00 PM" -> "11/30/2025"
    start_clean = start_raw.str.split().str[0]

    df_f["Date"] = pd.to_datetime(start_clean, errors="coerce").dt.date

    # Internal end time (for signature) will come from PDF-derived times only
    df_f["_End_dt"] = pd.NaT
    df_f["_ParentSig_dt"] = pd.to_datetime(df_f["Parent signature time"], errors="coerce")
    df_f["_UserSig_dt"] = pd.to_datetime(df_f["User signature time"], errors="coerce")

    # ---------- Normalize names ----------
    df_f["Staff Name"] = df_f["User"].apply(normalize_name)
    df_f["Client Name"] = df_f["Client"].apply(normalize_name)

    # ---------- Optional: Match to external Client/Session Time list ----------
    if external_sessions_df is not None:
        ext_df = external_sessions_df
        ext_df = normalize_cols(ext_df)
        if ensure_cols(ext_df, ["Client", "Session Time"], "External Sessions File"):
            # Normalize client names in the external file the same way
            ext_df["Client Name"] = ext_df["Client"].apply(normalize_name)

            # For each client, assign an order index (0,1,2,...) in the external file
            ext_df["SessionIndex"] = ext_df.groupby("Client Name").cumcount()

            # In the HiRasmus dataframe, assign per-client order index
            df_f["SessionIndex"] = df_f.groupby("Client Name").cumcount()

            # Merge on (Client Name, SessionIndex) WITHOUT changing df_f order
            df_f = df_f.merge(
                ext_df[["Client Name", "SessionIndex", "Session Time"]],
                on=["Client Name", "SessionIndex"],
                how="left",
                sort=False,
            )

            # True if this HiRasmus row has a matching external session record
            df_f["Has External Session"] = df_f["Session Time"].notna()

            # ---- Use Session Time to derive internal start/end + override duration ----
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
                "External sessions data provided, but required columns are missing."
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

    # ---------- Daily total minutes per Staff per Date ----------
    df_f["Daily Minutes"] = (
        df_f.groupby(["Staff Name", "Date"])["Actual Minutes"].transform("sum")
    )

    # ---------- Run Checks ----------
    eval_results = df_f.apply(evaluate_row, axis=1, result_type="expand")

    if isinstance(eval_results, pd.DataFrame):
        df_f["_DurationOk"] = eval_results["duration_ok"]
        df_f["_SigOk"] = eval_results["sig_ok"]
        df_f["_ExtOk"] = eval_results["ext_ok"]
        df_f["_DailyOk"] = eval_results["daily_ok"]
        df_f["_HasTimeAdjSig"] = eval_results["has_time_adj_sig"]
        df_f["_OverallPass"] = eval_results["overall_pass"]
    else:
        df_f["_DurationOk"] = eval_results.apply(lambda r: r["duration_ok"])
        df_f["_SigOk"] = eval_results.apply(lambda r: r["sig_ok"])
        df_f["_ExtOk"] = eval_results.apply(lambda r: r["ext_ok"])
        df_f["_DailyOk"] = eval_results.apply(lambda r: r["daily_ok"])
        df_f["_HasTimeAdjSig"] = eval_results.apply(lambda r: r["has_time_adj_sig"])
        df_f["_OverallPass"] = eval_results.apply(lambda r: r["overall_pass"])

    df_f["Failure Reasons"] = df_f.apply(get_failure_reasons, axis=1)

    # ---------- Pretty-print signature columns (show date + time) ----------
    for col in [
        "Parent signature time",
        "User signature time",
    ]:
        if col in df_f.columns:
            df_f[col] = (
                pd.to_datetime(df_f[col], errors="coerce")
                .dt.strftime("%m/%d/%Y %I:%M:%S %p")  # e.g. 11/29/2025 10:38:28 PM
                .fillna("")
            )


    # ---------- Display ----------
    display_cols = [
        "AlohaABA Appointment ID",
        "Date",
        "Client Name",
        "Staff Name",
        "Phone",
        "Email",
        "Duration",
        "Actual Minutes",
        "Daily Minutes",
        "Parent signature time",
        "User signature time",
        "Session Time",
    ]

    # Show the time-adjustment approval column if present
    if TIME_ADJ_COL in df_f.columns:
        display_cols.append(TIME_ADJ_COL)

    display_cols.extend(
        [
            "_DurationOk",
            "_SigOk",
            "_ExtOk",
            "_DailyOk",
            "_HasTimeAdjSig",
            "Failure Reasons",
        ]
    )

    present_cols = [c for c in display_cols if c in df_f.columns]

    st.subheader("2) Results")
    with st.expander("Summary", expanded=False):
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

    # ---------- Export (filter out 'Marry Wang Demo') ----------
    if "Client Name" in df_f.columns:
        export_df = df_f[df_f["Client Name"] != "Marry Wang Demo"].copy()
    else:
        export_df = df_f.copy()

    # Store final Session Checker dataframe for use in Tab 3 (Billed Checker)
    st.session_state["session_checker_df"] = export_df.copy()
    st.session_state["session_checker_present_cols"] = present_cols

    xlsx_all = export_excel(export_df[present_cols])
    xlsx_clean = export_excel(export_df[export_df["_OverallPass"]][present_cols])
    xlsx_flagged = export_excel(export_df[~export_df["_OverallPass"]][present_cols])

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

# =========================================================
# TAB 3: BILLED CHECKER – BILLED / UNBILLED / NO MATCH
# =========================================================
with tab3:
    st.header("Billed Checker – Billed / Unbilled / No Match")

    base_df = st.session_state.get("session_checker_df")
    present_cols = st.session_state.get("session_checker_present_cols")

    if base_df is None or present_cols is None:
        st.info(
            "Please run the **Session Checker** in Tab 2 first. "
            "Once you have results there, come back here."
        )
    else:
        st.subheader("Upload Aloha Billing Status File")

        billing_file = st.file_uploader(
            "Upload Aloha file with billed & unbilled sessions "
            "(must include 'Appointment ID' and 'Date Billed')",
            type=["xlsx", "xls", "csv"],
            key="billing_status_file",
        )

        if billing_file is not None:
            try:
                billing_df = read_any(billing_file)
            except Exception as e:
                billing_df = None
                st.error(f"Error reading billing status file: {e}")

            if billing_df is not None:
                billing_df = normalize_cols(billing_df)

                # Check required columns
                if "Appointment ID" not in billing_df.columns or "Date Billed" not in billing_df.columns:
                    st.error(
                        "Billing status file must contain columns: 'Appointment ID' and 'Date Billed'."
                    )
                elif "AlohaABA Appointment ID" not in base_df.columns:
                    st.error(
                        "Session Checker data is missing 'AlohaABA Appointment ID'. "
                        "Please ensure the HiRasmus export includes this column."
                    )
                else:
                    # Merge Session Checker data with billing status on Appointment ID
                    merged = base_df.merge(
                        billing_df[["Appointment ID", "Date Billed"]],
                        left_on="AlohaABA Appointment ID",
                        right_on="Appointment ID",
                        how="left",
                        sort=False,
                    )

                    # Classify billing status per row
                    def classify_status(row):
                        app_id = str(row.get("Appointment ID", "")).strip()
                        date_billed = str(row.get("Date Billed", "")).strip()

                        # No match at all in billing file
                        if app_id == "" or app_id.lower() == "nan":
                            return "No Match"

                        # Matched, now check if billed vs unbilled
                        if date_billed != "" and date_billed.lower() != "nan":
                            return "Billed"
                        return "Unbilled"

                    merged["Billing Status"] = merged.apply(classify_status, axis=1)

                    # Split into categories
                    billed_df = merged[merged["Billing Status"] == "Billed"]
                    unbilled_df = merged[merged["Billing Status"] == "Unbilled"]
                    nomatch_df = merged[merged["Billing Status"] == "No Match"]

                    # Split unbilled into clean vs flagged based on _OverallPass
                    unbilled_clean_df = unbilled_df[unbilled_df["_OverallPass"] == True]
                    unbilled_flagged_df = unbilled_df[unbilled_df["_OverallPass"] == False]

                    # Counts
                    total_sessions = len(merged)
                    billed_count = len(billed_df)
                    unbilled_clean_count = len(unbilled_clean_df)
                    unbilled_flagged_count = len(unbilled_flagged_df)
                    nomatch_count = len(nomatch_df)

                    # Summary
                    st.markdown(
                        f"""
**Total sessions (from Session Checker):** {total_sessions}  
- **Billed:** {billed_count}  
- **Unbilled – Clean:** {unbilled_clean_count}  
- **Unbilled – Flagged:** {unbilled_flagged_count}  
- **No Match (no Appointment ID match):** {nomatch_count}
"""
                    )

                    summary_df = pd.DataFrame(
                        {
                            "Status": [
                                "Billed",
                                "Unbilled – Clean",
                                "Unbilled – Flagged",
                                "No Match",
                            ],
                            "Count": [
                                billed_count,
                                unbilled_clean_count,
                                unbilled_flagged_count,
                                nomatch_count,
                            ],
                        }
                    )
                    st.table(summary_df)

                    # ---------- Downloads by Billing Status ----------
                    dl_cols = [
                        c
                        for c in dict.fromkeys(
                            present_cols
                            + ["Appointment ID", "Date Billed", "Billing Status"]
                        ).keys()
                        if c in merged.columns
                    ]

                    billed_dl = billed_df[dl_cols]
                    unbilled_clean_dl = unbilled_clean_df[dl_cols]
                    unbilled_flagged_dl = unbilled_flagged_df[dl_cols]
                    nomatch_dl = nomatch_df[dl_cols]

                    xlsx_billed = export_excel(billed_dl)
                    xlsx_unbilled_clean = export_excel(unbilled_clean_dl)
                    xlsx_unbilled_flagged = export_excel(unbilled_flagged_dl)
                    xlsx_nomatch = export_excel(nomatch_dl)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.download_button(
                            "⬇️ Billed Sessions",
                            data=xlsx_billed,
                            file_name="billed_sessions.xlsx",
                        )
                    with c2:
                        st.download_button(
                            "⬇️ Unbilled – Clean",
                            data=xlsx_unbilled_clean,
                            file_name="unbilled_clean_sessions.xlsx",
                        )
                    with c3:
                        st.download_button(
                            "⬇️ Unbilled – Flagged",
                            data=xlsx_unbilled_flagged,
                            file_name="unbilled_flagged_sessions.xlsx",
                        )
                    with c4:
                        st.download_button(
                            "⬇️ No Match Sessions",
                            data=xlsx_nomatch,
                            file_name="no_match_sessions.xlsx",
                        )
