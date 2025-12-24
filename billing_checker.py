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

MIN_MINUTES = 53    # >= 1 hour
MAX_MINUTES = 360   # <= 6 hours
BILLING_TOL_DEFAULT = 8      # up to 8 min over MAX allowed
DAILY_MAX_MINUTES = 480     # <= 8 hours per BT per day

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

DATE_RE = r"(\d{1,4}/\d{1,2}/\d{1,4})"
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


def normalize_session_time(raw: str) -> str:
    raw = re.sub(r"\b(a\.m\.|am)\b", "AM", raw, flags=re.I)
    raw = re.sub(r"\b(p\.m\.|pm)\b", "PM", raw, flags=re.I)

    if not raw:
        return ""

    raw = raw.strip()

    # Chinese Normalization
    if "上午" in raw or "下午" in raw:
        raw = raw.replace("上午", "AM ")
        raw = raw.replace("下午", "PM ")
        raw = re.sub(r"[–—]", "-", raw)
        raw = re.sub(r"\s+", " ", raw).strip()

    # Already 12h with AM/PM
    if re.search(r"\b(AM|PM)\b", raw, re.IGNORECASE):
        parts = re.split(r"\s*-\s*", raw)
        if len(parts) == 2:
            return f"{parts[0].strip()} - {parts[1].strip()}"
        return raw

    # 24h range → convert
    m = re.match(
        r"^\s*([0-9]{1,2}):([0-9]{2})\s*-\s*([0-9]{1,2}):([0-9]{2})\s*$",
        raw,
    )
    if not m:
        return raw

    h1, m1, h2, m2 = map(int, m.groups())

    def to_12h(h, minute):
        ampm = "AM"
        if h == 0:
            h12 = 12
        elif h < 12:
            h12 = h
        elif h == 12:
            h12 = 12
            ampm = "PM"
        else:
            h12 = h - 12
            ampm = "PM"
        return f"{h12}:{minute:02d} {ampm}"

    return f"{to_12h(h1, m1)} - {to_12h(h2, m2)}"


def normalize_date(raw: str) -> str:
    """
    Accepts:
      - 2025/10/28  (YYYY/MM/DD)
      - 10/28/2025  (MM/DD/YYYY)
    Returns normalized 'YYYY/MM/DD' when possible, else the stripped raw string.
    """
    if not raw:
        return ""

    raw = raw.strip()
    m = re.match(r"^(\d{1,4})/(\d{1,2})/(\d{1,4})$", raw)
    if not m:
        return raw

    a, b, c = m.groups()
    a_i, b_i, c_i = int(a), int(b), int(c)

    if len(a) == 4:           # YYYY/MM/DD
        year, month, day = a_i, b_i, c_i
    elif len(c) == 4:         # MM/DD/YYYY
        year, month, day = c_i, a_i, b_i
    else:                      # fallback
        year, month, day = c_i, a_i, b_i

    return f"{year:04d}/{month:02d}/{day:02d}"


def parse_notes(text: str):
    """
    Parse ABA session notes from extracted PDF text and enforce compliance.
    NOTE: Attendance is enforced here so Tab 2 can rely on Note Parse PASS only.
    """
    blocks = re.split(r"(?=Client\s*:)", text)
    results = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # =====================
        # BASIC IDENTIFIERS
        # =====================

        # IMPORTANT FIX: do NOT stop at comma; many PDFs show "Last, First"
        client_match = re.search(r"Client\s*:\s*([^\n]+)", block)
        client_name = client_match.group(1).strip() if client_match else ""
        if not client_name:
            continue

        provider_match = re.search(r"Rendering Provider\s*:\s*([^\n]+)", block)
        provider = provider_match.group(1).strip() if provider_match else ""

        # =====================
        # REQUIRED DEMOGRAPHICS
        # =====================

        dob_match = re.search(rf"Date of Birth\s*:\s*{DATE_RE}", block, re.I)
        dob = normalize_date(dob_match.group(1)) if dob_match else ""


        gender_match = re.search(r"Gender\s*:\s*([^\n\r]+)", block, re.I)
        gender_raw = gender_match.group(1).strip() if gender_match else ""

        # Normalize gender values
        g = gender_raw.strip().lower()
        if g in ("male", "m", "man", "boy", "男性", "男"):
            gender = "Male"
        elif g in ("female", "f", "woman", "girl", "女性", "女"):
            gender = "Female"
        else:
            gender = gender_raw  # keep whatever is provided

        # Diagnosis Code (ICD / IDC tolerant)
        diagnosis = ""
        dx_match = re.search(
            r"Diagnosis Code\s*\(\s*(?:ICD|IDC)\s*[-\s]*10\s*\)\s*:\s*([A-Za-z0-9\.\s]+)",
            block,
            re.I,
        )
        if dx_match:
            raw = dx_match.group(1)
            icd_match = re.search(r"[A-Za-z]\d{2}(?:\.\d+)?", raw, re.I)
            if icd_match:
                diagnosis = icd_match.group(0).upper()

        insurance_match = re.search(r"Primary Insurance\s*:\s*([^\n]+)", block)
        primary_insurance = insurance_match.group(1).strip() if insurance_match else ""

        ins_id_match = re.search(r"Insurance ID\s*:\s*([A-Z0-9]+)", block, re.I)
        insurance_id = ins_id_match.group(1).strip() if ins_id_match else ""

        # =====================
        # SESSION DATE & TIME
        # =====================

        date_match = re.search(rf"Session Date\s*:\s*{DATE_RE}", block, re.I)
        session_date = normalize_date(date_match.group(1)) if date_match else ""


        session_time_match = re.search(
            r"Session Time\s*:\s*([0-9]{1,2}:[0-9]{2}\s*(?:AM|PM)?\s*-\s*[0-9]{1,2}:[0-9]{2}\s*(?:AM|PM)?)",
            block,
            re.I,
        )
        raw_session_time = session_time_match.group(1) if session_time_match else ""

        if not raw_session_time:
            cn_match = re.search(
                r"(上午|下午)\s*[0-9]{1,2}:[0-9]{2}\s*-\s*(上午|下午)\s*[0-9]{1,2}:[0-9]{2}",
                block,
            )
            if cn_match:
                raw_session_time = cn_match.group(0)

        session_time = normalize_session_time(raw_session_time) if raw_session_time else ""

        # =====================
        # SESSION LOCATION
        # =====================

        location_match = re.search(r"Session Location\s*:\s*([^\n]+)", block)
        session_location = location_match.group(1).strip() if location_match else ""

        # =====================
        # PRESENT AT SESSION
        # =====================

        present_text = ""
        pos = block.lower().find("present at session")
        if pos != -1:
            present_text = block[pos : pos + 400]

        present_client = bool(re.search(r"\bClient\b", present_text, re.I))
        present_bt = bool(re.search(r"\b(BT/RBT|RBT/BT)\b", present_text, re.I))
        present_caregiver = bool(re.search(r"\bAdult Caregiver\b", present_text, re.I))
        present_sibling = bool(re.search(r"\bSibling(s)?\b", present_text, re.I))

        # =====================
        # MALADAPTIVE STATUS
        # =====================

        maladaptive_section = ""
        section_match = re.search(
            r"Maladaptive Status\s*:\s*(.*?)(?:\n[A-Z][a-zA-Z ]+?:|\Z)",
            block,
            re.S,
        )
        if section_match:
            maladaptive_section = section_match.group(1).strip()

        maladaptive_behaviors = []
        if maladaptive_section:
            for line in maladaptive_section.splitlines():
                clean = line.strip()
                if not clean:
                    continue

                lower = clean.lower()
                if (
                    lower.endswith(":")
                    or "continues to display" in lower
                    or "in the following areas" in lower
                    or "maladaptive status" in lower
                    or "other maladaptive behaviors" in lower
                ):
                    continue

                clean = re.sub(r"[•▪◦\-–—\uf0b7\uf0a7]+", "", clean).strip()

                if len(clean.split()) > 6 and not clean.lower().startswith("other"):
                    continue

                maladaptive_behaviors.append(clean.lower())

        other_selected = any(b == "other" or b.startswith("other ") for b in maladaptive_behaviors)

        other_desc_match = re.search(r"Other maladaptive behaviors\s*:\s*(.+)", block, re.I)
        other_maladaptive_present = bool(other_desc_match and other_desc_match.group(1).strip())

        # =====================
        # SESSION DATA CHECK
        # =====================

        data_rows = re.findall(
            r"\n\s*([a-zA-Z][a-zA-Z\s]+?)\s+([0-9]+)\s+([0-9]+)",
            block,
        )
        data_collected = len(data_rows) > 0

        # =====================
        # ATTESTATION & SIGNATURE
        # =====================

        provider_signature_present = bool(
            re.search(r"Provider\s+Signatures/Credentials\s+and\s+Date\s*:", block, re.I)
        )

        provider_signature_valid = bool(
            re.search(
                r"Provider\s+Signatures/Credentials\s+and\s+Date\s*:\s*"
                r"[A-Za-z][A-Za-z\s\-']+,\s*(?:BT|RBT)\s*,\s*"
                r"[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}",
                block,
                re.I,
            )
        )

        bt_attestation_present = bool(
            re.search(
                r"\battest\s+that\s+the\s+session\s+summary\s+is\s+accurate\s+and\s+correct\b",
                block,
                re.I,
            )
        )

        revision_attestation_present = bool(
            re.search(
                r"I\s+attest\s+the\s+revision/edit\s+made\s+to\s+this\s+note\s+as\s+signed\s+below\s+is\s+accurate\s+and\s+true",
                block,
                re.I,
            )
        )

        outcome_yes = bool(re.search(r"Outcome of Treatment.*?:\s*Yes", block, re.I | re.S))

        # =====================
        # COMPLIANCE VALIDATION
        # =====================

        compliance_errors = []

        if not dob:
            compliance_errors.append("Missing DOB")
        if not gender:
            compliance_errors.append("Missing Gender")
        if not diagnosis:
            compliance_errors.append("Missing ICD-10")
        if diagnosis and len(diagnosis) < 3:
            compliance_errors.append("Invalid ICD-10 code (truncated)")
        if not primary_insurance:
            compliance_errors.append("Missing Primary Insurance")
        if not insurance_id:
            compliance_errors.append("Missing Insurance ID")
        if not session_time:
            compliance_errors.append("Missing Session Time")
        if not session_location:
            compliance_errors.append("Missing Session Location")

        if not maladaptive_behaviors:
            compliance_errors.append("No maladaptive behaviors listed")

        if other_selected and not other_maladaptive_present:
            compliance_errors.append("Other maladaptive behavior selected but no description provided")

        if not data_collected:
            compliance_errors.append("No measurable session data found")

        if not outcome_yes:
            compliance_errors.append("Outcome of Treatment not Yes")

        if not bt_attestation_present:
            compliance_errors.append("Missing BT/RBT attestation statement")

        if not provider_signature_present:
            compliance_errors.append("Missing provider signature section")
        elif not provider_signature_valid:
            compliance_errors.append("Provider signature present but invalid format (must include Name, BT/RBT, and date)")

        # Attendance requirements (now enforced here)
        if not present_client:
            compliance_errors.append("Attendance: Client not present")
        if not present_bt:
            compliance_errors.append("Attendance: BT/RBT not present")
        if not (present_caregiver or present_sibling):
            compliance_errors.append("Attendance: Parent/Caregiver or Sibling not present")

        results.append(
            {
                "Client": client_name,
                "Rendering Provider": provider,
                "Session Date": session_date,
                "Date of Birth": dob,
                "Gender": gender,
                "Diagnosis Code": diagnosis,
                "Primary Insurance": primary_insurance,
                "Insurance ID": insurance_id,
                "Session Time": session_time,
                "Session Location": session_location,
                "Present_Client": present_client,
                "Present_BT_RBT": present_bt,
                "Present_Adult_Caregiver": present_caregiver,
                "Present_Sibling": present_sibling,
                "Maladaptive Behaviors": maladaptive_behaviors,
                "Other Selected": other_selected,
                "Other Maladaptive Provided": other_maladaptive_present,
                "Outcome Yes": outcome_yes,
                "Data Collected": data_collected,
                "BT Attestation Present": bt_attestation_present,
                "Provider Signature Present": provider_signature_present,
                "Provider Signature Valid": provider_signature_valid,
                "Revision Attestation Present": revision_attestation_present,
                "Compliance Errors": compliance_errors,
                "PASS": len(compliance_errors) == 0,
            }
        )

    return results


def notes_to_excel_bytes(results, sheet_name="Notes") -> bytes:
    """Convert list[dict] → Excel bytes."""
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
    Normalize names into: 'Last, First Middle' (proper-cased).
    Handles already-comma format like 'Last, First'.
    """
    if pd.isna(name):
        return ""
    s = str(name).strip()
    if not s:
        return ""

    def proper_case_block(block: str) -> str:
        block = " ".join(block.split())
        return " ".join(w.capitalize() for w in block.split())

    if "," in s:
        last, rest = [p.strip() for p in s.split(",", 1)]
        last = proper_case_block(last)
        rest = proper_case_block(rest)
        return f"{last}, {rest}" if rest else last

    parts = s.split()
    if len(parts) == 1:
        return proper_case_block(parts[0])

    last_raw = parts[-1]
    first_middle_raw = " ".join(parts[:-1])
    last = proper_case_block(last_raw)
    first_middle = proper_case_block(first_middle_raw)
    return f"{last}, {first_middle}"


def parse_session_time_range(session_time: Any, base_date: Any):
    """
    Parse '3:00 PM - 9:00 PM' into (start_dt, end_dt), using base_date.
    """
    if pd.isna(session_time):
        return pd.NaT, pd.NaT

    text = str(session_time).replace("\xa0", " ").strip()
    if not text or "-" not in text:
        return pd.NaT, pd.NaT

    start_str, end_str = [part.strip() for part in text.split("-", 1)]

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
        value=1440,
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
    DAILY_TOL = st.number_input(
        "Daily 8-hour limit tolerance (minutes)",
        value=8,
        min_value=0,
        step=1,
        help="How many extra minutes a BT can work over 8 hours in one day before it is flagged.",
    )

# BT Contacts (optional)
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

There are **three tabs** in this app:

- **Tools – Extract External Sessions**: Upload the *Session Notes PDF* from HiRasmus.  
  The app parses it and builds an **External Session List** and stores it for use in the Session Checker.

- **Session Checker**: Upload the HiRasmus sessions Excel export and run all compliance checks.  
  **Notes are REQUIRED**. If a note fails parsing compliance, the session fails here.

- **Billed Checker**: Upload Aloha billing status and see billed/unbilled/no-match.

"""
    )

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "1️⃣ Tools – Extract External Sessions",
        "2️⃣ Session Checker",
        "3️⃣ Billed Checker",
    ]
)

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
            text = pdf_bytes_to_text(pdf_file.read(), preserve_layout=True)
            results = parse_notes(text)

            if not results:
                st.warning("No notes found in the PDF. Make sure it contains 'Client:' blocks.")
            else:
                notes_df = pd.DataFrame(results)
                st.success(f"Parsed {len(notes_df)} notes from PDF.")

                with st.expander("Preview parsed notes", expanded=False):
                    st.dataframe(notes_df.head(50))

                cols = [
                    "Client",
                    "Session Time",
                    "Present_Client",
                    "Present_Adult_Caregiver",
                    "Present_Sibling",
                    "Present_BT_RBT",
                    "PASS",
                    "Compliance Errors",
                ]
                existing_cols = [c for c in cols if c in notes_df.columns]

                ext_df = notes_df[existing_cols].copy()
                ext_df = ext_df.rename(
                    columns={
                        "PASS": "Note Parse PASS",
                        "Compliance Errors": "Note Compliance Errors",
                    }
                )

                ext_df = normalize_cols(ext_df)
                ext_df = ext_df[ext_df["Client"].notna()]

                st.session_state["external_sessions_df"] = ext_df

                st.info(
                    f"Generated {len(ext_df)} external session rows. "
                    "These will be used automatically in the **Session Checker** tab."
                )

                ext_xlsx = notes_to_excel_bytes(
                    ext_df.to_dict(orient="records"),
                    sheet_name="External Sessions",
                )
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
def has_time_adjust_sig(row) -> bool:
    """True if time-adjustment parent approval signature exists and is non-empty."""
    if TIME_ADJ_COL not in row.index:
        return False
    val = row.get(TIME_ADJ_COL)
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s not in ("", "nan")


def duration_ok_base(row) -> bool:
    """Duration check with over-max tolerance; no tolerance below MIN."""
    m = row.get("Actual Minutes")
    if pd.isna(m):
        return False
    if m < MIN_MINUTES:
        return False
    if m <= MAX_MINUTES:
        return True
    return (m - MAX_MINUTES) <= BILLING_TOL


def external_time_ok(row) -> bool:
    """
    External session time must exist AND be parseable into start/end.
    Notes are required; if not present, fail.
    """
    if "Has External Session" not in row.index:
        return False
    if not bool(row.get("Has External Session")):
        return False
    return (not pd.isna(row.get("_ExtStart_dt"))) and (not pd.isna(row.get("_ExtEnd_dt")))


def sig_ok_base(row) -> bool:
    """
    Signature timing check against session end time (_End_dt).
    If both signatures are missing, treat as OK.
    If signatures exist but no _End_dt, fail.
    """
    base_ts = row.get("_End_dt", pd.NaT)
    parent_sig_ts = row.get("_ParentSig_dt", pd.NaT)
    user_sig_ts = row.get("_UserSig_dt", pd.NaT)

    if pd.isna(parent_sig_ts) and pd.isna(user_sig_ts):
        return True

    if pd.isna(base_ts):
        return False

    checks = []
    if not pd.isna(parent_sig_ts):
        checks.append(within_time_tol(parent_sig_ts, base_ts, SIG_TOL_EARLY, SIG_TOL_LATE))
    if not pd.isna(user_sig_ts):
        checks.append(within_time_tol(user_sig_ts, base_ts, SIG_TOL_EARLY, SIG_TOL_LATE))

    return all(checks) if checks else False


def daily_total_ok(row) -> bool:
    """Daily total minutes per staff per day must be under cap + tolerance."""
    m = row.get("Daily Minutes")
    if pd.isna(m):
        return True
    return m < (DAILY_MAX_MINUTES + DAILY_TOL)


def note_parse_ok(row) -> bool:
    """Notes must have everything; require note parse PASS."""
    val = row.get("_NoteParseOk")
    return bool(val) if not pd.isna(val) else False


def evaluate_row(row) -> dict:
    dur_base = duration_ok_base(row)
    sig_base = sig_ok_base(row)
    ext_ok = external_time_ok(row)
    adj_sig = has_time_adjust_sig(row)
    daily_ok_val = daily_total_ok(row)
    note_parse_ok_val = note_parse_ok(row)

    duration_ok = dur_base
    sig_ok = sig_base

    # If override is enabled and the time-adjustment signature exists, ignore signature timing failures
    if USE_TIME_ADJ_OVERRIDE and adj_sig:
        sig_ok = True

    overall = (
        duration_ok
        and sig_ok
        and daily_ok_val
        and ext_ok
        and note_parse_ok_val
    )

    return {
        "duration_ok": duration_ok,
        "sig_ok": sig_ok,
        "ext_ok": ext_ok,
        "daily_ok": daily_ok_val,
        "note_parse_ok": note_parse_ok_val,
        "has_time_adj_sig": adj_sig,
        "overall_pass": overall,
        "duration_ok_base": dur_base,
        "sig_ok_base": sig_base,
    }


def get_failure_reasons(row) -> str:
    eval_res = evaluate_row(row)
    reasons = []

    # Note parse failures FIRST (most important)
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

    # External time failures
    if not eval_res["ext_ok"]:
        stime = row.get("Session Time", "")
        if pd.isna(stime) or str(stime).strip() == "":
            reasons.append("Session Time missing/blank (cannot derive start/end)")
        else:
            reasons.append("Session Time present but could not be parsed into start/end")

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

    # Signature failures (after override)
    if not eval_res["sig_ok"]:
        tol_str = f"{SIG_TOL_EARLY}/+{SIG_TOL_LATE}"
        if USE_TIME_ADJ_OVERRIDE and not eval_res["has_time_adj_sig"]:
            reasons.append(
                f"Signature not within {tol_str} minutes of end time (no time-adjustment override)"
            )
        else:
            reasons.append(f"Signature not within {tol_str} minutes of end time")

    # Daily total failures
    if not eval_res.get("daily_ok", True):
        daily_min = row.get("Daily Minutes")
        if not pd.isna(daily_min):
            reasons.append(
                f"Total daily duration for this BT on {row.get('Date')} "
                f"({daily_min:.0f} min) exceeds {DAILY_MAX_MINUTES} min + {DAILY_TOL} min tolerance"
            )

    return "; ".join(reasons) if reasons else "PASS"


with tab2:
    st.header("Session Checker")

    st.subheader("Upload HiRamsus Excel File")
    sessions_file = st.file_uploader(
        "Upload HiRamsus Excel File",
        type=["xlsx", "xls"],
        key="sessions_file",
    )

    if not sessions_file:
        st.info("Upload the HiRamsus Excel file to continue.")
        st.stop()

    # Notes are REQUIRED
    external_sessions_df = st.session_state.get("external_sessions_df", None)
    if external_sessions_df is None or len(external_sessions_df) == 0:
        st.error("Session Notes PDF is REQUIRED. Please upload and parse notes in Tab 1 first.")
        st.stop()

    st.success(f"Using {len(external_sessions_df)} external session rows from the **Tools** tab.")

    # Checkbox: use second parent signature as override for signature timing only
    global USE_TIME_ADJ_OVERRIDE
    USE_TIME_ADJ_OVERRIDE = st.toggle(
        f"Use '{TIME_ADJ_COL}' as a signature override (signature timing only; duration & daily limit still enforced)",
        value=False,
    )

    # ---------- Read and normalize Sessions file ----------
    df = pd.read_excel(sessions_file, dtype=object)
    df = normalize_cols(df)

    if "Start date" not in df.columns and "Start Date" in df.columns:
        df = df.rename(columns={"Start Date": "Start date"})

    if not ensure_cols(df, REQ_COLS, "Sessions File"):
        st.stop()

    if USE_TIME_ADJ_OVERRIDE:
        if not ensure_cols(df, [TIME_ADJ_COL], "Sessions File"):
            st.stop()

    # ---------- Prefilter by Status + Session ----------
    df_f = df[
        (df["Status"].astype(str).str.strip() == STATUS_REQUIRED)
        & (df["Session"].astype(str).str.strip() == SESSION_REQUIRED)
    ].copy()

    # ---------- Parse Duration ----------
    df_f["Actual Minutes"] = df_f["Duration"].apply(parse_duration_to_minutes)

    # ---------- Parse Date from Start date ----------
    start_raw = df_f["Start date"].astype(str).str.strip()
    start_clean = start_raw.str.split().str[0]
    df_f["Date"] = pd.to_datetime(start_clean, errors="coerce").dt.date

    # Signature timestamps
    df_f["_End_dt"] = pd.NaT
    df_f["_ParentSig_dt"] = pd.to_datetime(df_f["Parent signature time"], errors="coerce")
    df_f["_UserSig_dt"] = pd.to_datetime(df_f["User signature time"], errors="coerce")

    # ---------- Normalize names ----------
    df_f["Staff Name"] = df_f["User"].apply(normalize_name)
    df_f["Client Name"] = df_f["Client"].apply(normalize_name)

    # ---------- Match to external Client/Session Time list ----------
    ext_df = normalize_cols(external_sessions_df)

    if not ensure_cols(ext_df, ["Client", "Session Time", "Note Parse PASS", "Note Compliance Errors"], "External Sessions (from Tab 1)"):
        st.stop()

    ext_df["Client Name"] = ext_df["Client"].apply(normalize_name)
    ext_df["SessionIndex"] = ext_df.groupby("Client Name").cumcount()

    df_f = df_f.sort_values(by=["Client Name", "Date", "Start date"], na_position="last").reset_index(drop=True)
    df_f["SessionIndex"] = df_f.groupby("Client Name").cumcount()

    merge_cols = [
        "Client Name",
        "SessionIndex",
        "Session Time",
        "Present_Client",
        "Present_Adult_Caregiver",
        "Present_Sibling",
        "Present_BT_RBT",
        "Note Parse PASS",
        "Note Compliance Errors",
    ]
    merge_cols = [c for c in merge_cols if c in ext_df.columns]

    df_f = df_f.merge(
        ext_df[merge_cols],
        on=["Client Name", "SessionIndex"],
        how="left",
        sort=False,
    )

    # Note parse gate: if no match or parse fail => False
    df_f["_NoteParseOk"] = df_f["Note Parse PASS"].fillna(False).astype(bool)

    # Derive Has External Session from non-blank Session Time
    df_f["Has External Session"] = (
        df_f["Session Time"].notna()
        & (df_f["Session Time"].astype(str).str.strip() != "")
    )

    # Derive external start/end + set internal end
    df_f[["_ExtStart_dt", "_ExtEnd_dt"]] = df_f.apply(
        lambda r: pd.Series(parse_session_time_range(r.get("Session Time"), r.get("Date"))),
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
    df_f["Daily Minutes"] = df_f.groupby(["Staff Name", "Date"])["Actual Minutes"].transform("sum")

    # ---------- Run Checks ----------
    eval_results = df_f.apply(evaluate_row, axis=1, result_type="expand")

    df_f["_DurationOk"] = eval_results["duration_ok"]
    df_f["_SigOk"] = eval_results["sig_ok"]
    df_f["_ExtOk"] = eval_results["ext_ok"]
    df_f["_DailyOk"] = eval_results["daily_ok"]
    df_f["_HasTimeAdjSig"] = eval_results["has_time_adj_sig"]
    df_f["_NoteParseOk"] = eval_results["note_parse_ok"]
    df_f["_OverallPass"] = eval_results["overall_pass"]

    df_f["Failure Reasons"] = df_f.apply(get_failure_reasons, axis=1)

    # ---------- Pretty-print signature columns ----------
    for col in ["Parent signature time", "User signature time"]:
        if col in df_f.columns:
            df_f[col] = (
                pd.to_datetime(df_f[col], errors="coerce")
                .dt.strftime("%m/%d/%Y %I:%M:%S %p")
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
        "Note Parse PASS",
        "Note Compliance Errors",
    ]

    if TIME_ADJ_COL in df_f.columns:
        display_cols.append(TIME_ADJ_COL)

    display_cols.extend(
        [
            "_DurationOk",
            "_SigOk",
            "_ExtOk",
            "_DailyOk",
            "_HasTimeAdjSig",
            "_NoteParseOk",
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

    st.session_state["session_checker_df"] = export_df.copy()
    st.session_state["session_checker_present_cols"] = present_cols

    xlsx_all = export_excel(export_df[present_cols])
    xlsx_clean = export_excel(export_df[export_df["_OverallPass"]][present_cols])
    xlsx_flagged = export_excel(export_df[~export_df["_OverallPass"]][present_cols])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Download All", data=xlsx_all, file_name="all_sessions.xlsx")
    with c2:
        st.download_button("✅ Passed Only", data=xlsx_clean, file_name="clean_sessions.xlsx")
    with c3:
        st.download_button("⚠️ Failed Only", data=xlsx_flagged, file_name="flagged_sessions.xlsx")


# =========================================================
# TAB 3: BILLED CHECKER – BILLED / UNBILLED / NO MATCH
# =========================================================
with tab3:
    st.header("Billed Checker – Billed / Unbilled / No Match")

    base_df = st.session_state.get("session_checker_df")
    present_cols = st.session_state.get("session_checker_present_cols")

    if base_df is None or present_cols is None:
        st.info("Please run the **Session Checker** in Tab 2 first.")
    else:
        st.subheader("Upload Aloha Billing Status File")

        billing_file = st.file_uploader(
            "Upload Aloha file with billed & unbilled sessions (must include 'Appointment ID' and 'Date Billed')",
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

                if "Appointment ID" not in billing_df.columns or "Date Billed" not in billing_df.columns:
                    st.error("Billing status file must contain columns: 'Appointment ID' and 'Date Billed'.")
                elif "AlohaABA Appointment ID" not in base_df.columns:
                    st.error("Session Checker data is missing 'AlohaABA Appointment ID'.")
                else:
                    merged = base_df.merge(
                        billing_df[["Appointment ID", "Date Billed"]],
                        left_on="AlohaABA Appointment ID",
                        right_on="Appointment ID",
                        how="left",
                        sort=False,
                    )

                    cols_to_drop = [c for c in ["Client", "User"] if c in merged.columns]
                    merged = merged.drop(columns=cols_to_drop)

                    desired_order = [
                        "Status",
                        "Date/Time",
                        "End time",
                        "Duration",
                        "Activity type",
                        "Session",
                        "AlohaABA Appointment ID",
                        "Parent signature time",
                        "User signature time",
                        "Start date",
                        "Actual Minutes",
                        "Date",
                        "_End_dt",
                        "_ParentSig_dt",
                        "_UserSig_dt",
                        "Client Name",
                        "SessionIndex",
                        "Session Time",
                        "Note Parse PASS",
                        "Note Compliance Errors",
                        "Has External Session",
                        "_ExtStart_dt",
                        "_ExtEnd_dt",
                        "Phone",
                        "Email",
                        "Daily Minutes",
                        "_DurationOk",
                        "_SigOk",
                        "_ExtOk",
                        "_DailyOk",
                        "_HasTimeAdjSig",
                        "_NoteParseOk",
                        "_OverallPass",
                        "Failure Reasons",
                        "Appointment ID",
                        "Date Billed",
                        "Billing Status",
                    ]

                    ordered_cols = [c for c in desired_order if c in merged.columns]
                    merged = merged[ordered_cols]

                    def classify_status(row):
                        app_id = str(row.get("Appointment ID", "")).strip()
                        date_billed = str(row.get("Date Billed", "")).strip()

                        if app_id == "" or app_id.lower() == "nan":
                            return "No Match"
                        if date_billed != "" and date_billed.lower() != "nan":
                            return "Billed"
                        return "Unbilled"

                    merged["Billing Status"] = merged.apply(classify_status, axis=1)

                    billed_df = merged[merged["Billing Status"] == "Billed"]
                    unbilled_df = merged[merged["Billing Status"] == "Unbilled"]
                    nomatch_df = merged[merged["Billing Status"] == "No Match"]

                    unbilled_clean_df = unbilled_df[unbilled_df["_OverallPass"] == True]
                    unbilled_flagged_df = unbilled_df[unbilled_df["_OverallPass"] == False]

                    total_sessions = len(merged)
                    billed_count = len(billed_df)
                    unbilled_clean_count = len(unbilled_clean_df)
                    unbilled_flagged_count = len(unbilled_flagged_df)
                    nomatch_count = len(nomatch_df)

                    st.markdown(
                        f"""
**Total sessions (from Session Checker):** {total_sessions}  
- **Billed:** {billed_count}  
- **Unbilled – Clean:** {unbilled_clean_count}  
- **Unbilled – Flagged:** {unbilled_flagged_count}  
- **No Match:** {nomatch_count}
"""
                    )

                    summary_df = pd.DataFrame(
                        {
                            "Status": ["Billed", "Unbilled – Clean", "Unbilled – Flagged", "No Match"],
                            "Count": [billed_count, unbilled_clean_count, unbilled_flagged_count, nomatch_count],
                        }
                    )
                    st.table(summary_df)

                    dl_cols = [
                        c
                        for c in dict.fromkeys(present_cols + ["Appointment ID", "Date Billed", "Billing Status"]).keys()
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
                    xlsx_all = export_excel(merged[dl_cols])

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.download_button("⬇️ Billed Sessions", data=xlsx_billed, file_name="billed_sessions.xlsx")
                    with c2:
                        st.download_button("⬇️ Unbilled – Clean", data=xlsx_unbilled_clean, file_name="unbilled_clean_sessions.xlsx")
                    with c3:
                        st.download_button("⬇️ Unbilled – Flagged", data=xlsx_unbilled_flagged, file_name="unbilled_flagged_sessions.xlsx")
                    with c4:
                        st.download_button("⬇️ No Match Sessions", data=xlsx_nomatch, file_name="no_match_sessions.xlsx")
                    with c5:
                        st.download_button("⬇️ All Sessions", data=xlsx_all, file_name="all_sessions.xlsx")
