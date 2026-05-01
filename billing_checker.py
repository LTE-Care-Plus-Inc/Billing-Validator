# ===========================
# app.py
# ===========================
import io
import logging
import re
import time
from difflib import SequenceMatcher
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from constants import (
    STATUS_REQUIRED,
    SESSION_REQUIRED,
    BILLING_TOL_DEFAULT,
    TIME_ADJ_COL,
    REQ_COLS,
    MIN_SESSION_GAP_MINUTES,
)
from utils import (
    normalize_date,
    normalize_session_time,
    parse_session_time_range,
    normalize_time_range,
    read_any,
    normalize_cols,
    ensure_cols,
    parse_duration_to_minutes,
    normalize_client_name_for_match,
    normalize_name,
    export_excel,
)
from parsers import (
    pdf_bytes_to_text,
    parse_notes,
    notes_to_excel_bytes,
)
from validators import (
    evaluate_row,
    get_failure_reasons,
    configure as configure_validators,
)

_LOG = logging.getLogger("billing_checker")
if not _LOG.handlers:
    _stderr = logging.StreamHandler()
    _stderr.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _LOG.addHandler(_stderr)
_LOG.setLevel(logging.INFO)
_LOG.propagate = False


def _log_step(name: str, t0: float) -> None:
    _LOG.info("%s: %.4f s", name, time.perf_counter() - t0)


def _norm_bt_contact_key(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(",", " ")
    s = " ".join(s.split())
    return s


def _map_staff_to_contact_lookup(staff_series: pd.Series, contact_map: dict) -> pd.Series:
    """Resolve phone/email using str keys from the BT cache (pandas cells may be numpy string scalars)."""
    return staff_series.map(lambda x: contact_map.get(str(x), "") if pd.notna(x) else "")


@st.cache_data(show_spinner=False)
def _cached_bt_contact_maps(
    bt_blob: bytes,
    bt_filename: str,
    staff_tuple: Tuple[str, ...],
) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...], Optional[str]]:
    """Build staff→Phone/Email maps (SequenceMatcher ratio ≥ 0.8); same semantics as non-cached code.

    Cached by BT file contents + normalized staff roster so Streamlit reruns need not redo O(n×m) work.
    Returns ``(phone_item_pairs, email_item_pairs, error_or_none)``; caller wraps with dict(...).
    """
    buf = io.BytesIO(bt_blob)
    buf.name = bt_filename
    try:
        bt_df = read_any(buf)
    except Exception as e:
        return (), (), f"BT Contacts file could not be read: {e}"

    if bt_df is None:
        return (), (), "BT Contacts file was empty."

    bt_df = normalize_cols(bt_df)
    bt_required = {"BT Name", "Phone", "Email"}
    bt_missing = bt_required - set(bt_df.columns)
    if bt_missing:
        return (), (), f"BT Contacts file is missing: {sorted(bt_missing)}"

    bt_df["BT_formatted"] = bt_df["BT Name"].apply(normalize_name)
    bt_df["bt_norm"] = bt_df["BT_formatted"].apply(_norm_bt_contact_key)

    exact_phone_by_norm = {}
    exact_email_by_norm = {}
    for bt_norm, phone, email in zip(
        bt_df["bt_norm"],
        bt_df["Phone"],
        bt_df["Email"],
    ):
        if bt_norm not in exact_phone_by_norm:
            exact_phone_by_norm[bt_norm] = "" if pd.isna(phone) else str(phone)
            exact_email_by_norm[bt_norm] = "" if pd.isna(email) else str(email)

    bt_entries = list(zip(bt_df["bt_norm"], bt_df["Phone"], bt_df["Email"]))

    staff_to_phone = {}
    staff_to_email = {}
    for staff in staff_tuple:
        staff_norm = _norm_bt_contact_key(staff)

        if staff_norm in exact_phone_by_norm:
            staff_to_phone[staff] = exact_phone_by_norm[staff_norm]
            staff_to_email[staff] = exact_email_by_norm[staff_norm]
            continue

        best_score = 0.0
        best_phone = ""
        best_email = ""
        for bt_name_norm, phone, email in bt_entries:
            score = SequenceMatcher(None, staff_norm, bt_name_norm).ratio()
            if score > best_score:
                best_score = score
                best_phone = "" if pd.isna(phone) else str(phone)
                best_email = "" if pd.isna(email) else str(email)

        if best_score >= 0.8:
            staff_to_phone[staff] = best_phone
            staff_to_email[staff] = best_email

    return tuple(staff_to_phone.items()), tuple(staff_to_email.items()), None


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Session Compliance Checker",
    layout="wide",
)
st.title("Session Compliance Checker")

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("Settings")

with st.sidebar.expander("Signature Settings", expanded=False):
    SIG_TOL_EARLY = st.number_input(
        "Signature early tolerance (minutes, negative)",
        value=-8,
        step=1,
        help="Earliest allowed time before session end (e.g., -15 means 15 minutes before).",
    )

with st.sidebar.expander("Duration / Billing Settings", expanded=False):
    BILLING_TOL = st.number_input(
        "Billing duration tolerance (minutes)",
        value=BILLING_TOL_DEFAULT,
        min_value=0,
        step=1,
        help=(
            "If a session's duration is outside the base range by at most this many minutes, "
            "it will still be treated as OK."
        ),
    )
    DAILY_TOL = st.number_input(
        "Daily 8-hour limit tolerance (minutes)",
        value=8,
        min_value=0,
        step=1,
        help="How many extra minutes a BT can work over 8 hours in one day before it is flagged."
    )

st.sidebar.header("BT Contacts (optional)")
bt_contacts_file = st.sidebar.file_uploader(
    "Upload BT Contacts (BT Name, Phone, Email)",
    type=["csv", "xlsx"],
    key="bt_contacts",
)

st.sidebar.header("Zoho CRM (optional)")
zoho_file = st.sidebar.file_uploader(
    "Upload Zoho Export (for Case Coordinator Name)",
    type=["csv", "xlsx"],
    key="zoho_file",
    help="Must contain 'Medicaid ID', 'Date of Birth', and 'Case Coordinator Name' columns.",
)

with st.expander("Instructions", expanded=False):
    st.markdown(
        """
This tool reviews **BT 1:1 Direct Service sessions from HiRasmus** and checks whether each session meets
core billing-compliance requirements.

There are **three tabs** in this app:

- **Tools – Extract External Sessions**: Upload the *Session Notes PDF* from HiRasmus.
  The app parses it and builds an **External Session List** and stores it for use in the Session Checker.

- **Session Checker**: Upload the HiRasmus sessions Excel export and run all compliance checks.

- **Billed Checker**: Upload Aloha billing status and see billed/unbilled/no-match.
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

if "external_sessions_df" not in st.session_state:
    st.session_state["external_sessions_df"] = None

# =========================================================
# TAB 1
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
            _t = time.perf_counter()
            _pdf_bytes = pdf_file.read()
            text = pdf_bytes_to_text(_pdf_bytes, preserve_layout=True)
            results = parse_notes(text)
            _log_step("tab1.pdf_read_text_extract_and_parse_notes", _t)

            if not results:
                st.warning("No notes found in the PDF. Make sure it contains 'Client:' blocks.")
            else:
                _t = time.perf_counter()
                notes_df = pd.DataFrame(results)
                st.success(f"Parsed {len(notes_df)} notes from PDF.")

                with st.expander("Preview parsed notes (compliance)", expanded=False):
                    preview_cols = [
                        "Client",
                        "Session Date",
                        "Session Time",
                        "Session Location",
                        "PASS",
                        "Compliance Errors",
                    ]
                    preview_cols = [c for c in preview_cols if c in notes_df.columns]
                    st.dataframe(notes_df[preview_cols].head(120), width="stretch")

                cols = [
                    "Client",
                    "Session Time",
                    "Session Date",
                    "Insurance ID",
                    "Date of Birth",          # needed for DOB-fallback matching in Tab 2
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
                        "Present_Adult_Caregiver": "Present_ParentCaregiver",
                        "PASS": "Note Parse PASS",
                        "Compliance Errors": "Note Compliance Errors",
                    }
                )
                ext_df = normalize_cols(ext_df)
                ext_df = ext_df[ext_df["Client"].notna()].copy()

                st.session_state["external_sessions_df"] = ext_df
                _log_step(
                    "tab1.build_external_sessions_dataframe_normalize_and_store",
                    _t,
                )
                _LOG.info(
                    "tab1.done: external_sessions_df rows=%d (ready for Session Checker)",
                    len(ext_df),
                )

                st.info(
                    f"Generated {len(ext_df)} external session rows. "
                    "These will be used automatically in the **Session Checker** tab."
                )

                _t = time.perf_counter()
                ext_xlsx = notes_to_excel_bytes(ext_df.to_dict(orient="records"), sheet_name="External Sessions")
                _log_step("tab1.build_download_xlsx_buffer", _t)
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

    external_sessions_df = st.session_state.get("external_sessions_df", None)

    if external_sessions_df is not None and len(external_sessions_df) > 0:
        st.success(f"Using {len(external_sessions_df)} external session rows from the **Tools** tab.")
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
            _t = time.perf_counter()
            external_sessions_df = read_any(manual_external_file)
            _log_step("tab2.load_manual_external_sessions_file", _t)

    USE_TIME_ADJ_OVERRIDE = st.toggle(
        f"Use '{TIME_ADJ_COL}' as a signature override (only for sessions that failed signature timing; duration & daily limit still enforced)",
        value=True,
    )

    # Inject runtime settings into validators module
    configure_validators(
        billing_tol=BILLING_TOL,
        daily_tol=DAILY_TOL,
        sig_tol_early=SIG_TOL_EARLY,
        use_time_adj_override=USE_TIME_ADJ_OVERRIDE,
    )

    _tab2_pipeline_start = time.perf_counter()
    _t = time.perf_counter()
    df = pd.read_excel(sessions_file, dtype=object)
    df = normalize_cols(df)

    if "Start date" not in df.columns and "Start Date" in df.columns:
        df = df.rename(columns={"Start Date": "Start date"})

    if not ensure_cols(df, REQ_COLS, "Sessions File"):
        _LOG.info("tab2.aborted: sessions file missing required columns")
        st.stop()

    if USE_TIME_ADJ_OVERRIDE:
        if not ensure_cols(df, [TIME_ADJ_COL], "Sessions File"):
            _LOG.info("tab2.aborted: sessions file missing %s column", TIME_ADJ_COL)
            st.stop()

    _log_step("tab2.read_sessions_excel_and_validate_columns", _t)

    _t = time.perf_counter()
    df_f = df[
        (df["Status"].astype(str).str.strip().isin(STATUS_REQUIRED))
        & (df["Session"].astype(str).str.strip() == SESSION_REQUIRED)
        & (df["AlohaABA Appointment Type"].astype(str).str.strip() != "Supervision")
    ].copy()

    df_f["_RowOrder"] = np.arange(len(df_f))
    df_f["Actual Minutes"] = df_f["Duration"].apply(parse_duration_to_minutes)

    start_raw = df_f["Start date"].astype(str).str.strip()
    start_clean = start_raw.str.split().str[0]
    df_f["Date"] = pd.to_datetime(start_clean, errors="coerce").dt.strftime("%m/%d/%Y")

    df_f["_End_dt"] = pd.NaT
    df_f["_ParentSig_dt"] = (
        pd.to_datetime(df_f["Adult Caregiver signature time"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    df_f["Staff Name"] = df_f["User"].apply(normalize_client_name_for_match)
    df_f["Client Name"] = df_f["Client"].apply(normalize_client_name_for_match)

    # ---- Parse Excel session times (Date/Time = start, End time = end) ----
    df_f["_ExcelStart_dt"] = (
        pd.to_datetime(df_f["Date/Time"], errors="coerce")
        if "Date/Time" in df_f.columns else pd.NaT
    )
    df_f["_ExcelEnd_dt"] = (
        pd.to_datetime(df_f["End time"], errors="coerce")
        if "End time" in df_f.columns else pd.NaT
    )
    _log_step("tab2.filter_rows_and_prepare_timestamps", _t)

    # ---- External session matching ----
    if external_sessions_df is not None:
        ext_df = normalize_cols(external_sessions_df.copy())

        if ensure_cols(ext_df, ["Insurance ID", "Session Time"], "External Sessions File"):
            ext_df["Client Name"] = ext_df["Client"].apply(normalize_client_name_for_match)

            # Rename so both sides share the same column name
            ext_df = ext_df.rename(columns={"Insurance ID": "Client: Insurance ID"})

            # Payload columns to bring over from the PDF notes (not join keys)
            _data_cols = [c for c in [
                "Session Time",
                "Session Date",
                "Present_Client",
                "Present_ParentCaregiver",
                "Present_Sibling",
                "Present_BT_RBT",
                "Note Parse PASS",
                "Note Compliance Errors",
            ] if c in ext_df.columns]

            # ── PRIMARY MERGE: Insurance ID + positional index ───────────────
            _tp = time.perf_counter()
            ext_df["SessionIndex"] = ext_df.groupby("Client: Insurance ID").cumcount()
            df_f["SessionIndex"]   = df_f.groupby("Client: Insurance ID").cumcount()

            df_f = df_f.merge(
                ext_df[["Client: Insurance ID", "SessionIndex"] + _data_cols],
                on=["Client: Insurance ID", "SessionIndex"],
                how="left",
                sort=False,
            )

            # ── FALLBACK MERGE: DOB + positional index ───────────────────────
            # Only runs for rows that didn't get a Session Time from the primary merge.
            _unmatched = df_f["Session Time"].isna()
            _ext_has_dob = "Date of Birth" in ext_df.columns
            _df_has_dob  = "Client: Date of Birth" in df_f.columns

            if _unmatched.any() and _ext_has_dob and _df_has_dob:
                # Normalize to date-only (strip time component) for stable comparison
                ext_df["_dob_norm"] = pd.to_datetime(
                    ext_df["Date of Birth"], errors="coerce"
                ).dt.normalize()
                df_f["_dob_norm"] = pd.to_datetime(
                    df_f["Client: Date of Birth"], errors="coerce"
                ).dt.normalize()

                # Positional index within each DOB group (mirrors Insurance ID logic)
                ext_df["_SIdx_DOB"] = ext_df.groupby("_dob_norm").cumcount()
                df_f["_SIdx_DOB"]   = df_f.groupby("_dob_norm").cumcount()

                dob_result = (
                    df_f.loc[_unmatched, ["_dob_norm", "_SIdx_DOB"]]
                    .merge(
                        ext_df[["_dob_norm", "_SIdx_DOB"] + _data_cols],
                        on=["_dob_norm", "_SIdx_DOB"],
                        how="left",
                        sort=False,
                    )
                )
                for col in _data_cols:
                    if col in dob_result.columns:
                        df_f.loc[_unmatched, col] = dob_result[col].values

                df_f.drop(columns=["_dob_norm", "_SIdx_DOB"], errors="ignore", inplace=True)
                ext_df.drop(columns=["_dob_norm", "_SIdx_DOB"], errors="ignore", inplace=True)

            df_f = df_f.sort_values("_RowOrder", kind="mergesort").reset_index(drop=True)
            _log_step("tab2.merge_external_sessions_pandas_joins", _tp)

            # ---- MERGE DEBUGGING ----
            with st.expander("🔍 Merge Debug", expanded=True):
                _debug_mask = df_f["Client Name"].astype(str).str.lower().str.contains("demo|marry", na=False)
                _df_debug = df_f[~_debug_mask]

                total    = len(_df_debug)
                matched  = _df_debug["Session Time"].notna().sum()
                still_unmatched = total - matched
                st.write(
                    f"**Total rows:** {total} | "
                    f"**Matched (Insurance ID or DOB):** {matched} | "
                    f"**Still unmatched:** {still_unmatched}"
                )

                st.write("**ext_df Insurance IDs (sample):**", ext_df["Client: Insurance ID"].dropna().unique()[:10].tolist())
                st.write("**df_f Insurance IDs (sample):**", _df_debug["Client: Insurance ID"].dropna().unique()[:10].tolist())

                st.write("**SessionIndex range in ext_df:**", ext_df["SessionIndex"].min(), "–", ext_df["SessionIndex"].max())
                st.write("**SessionIndex range in df_f:**", _df_debug["SessionIndex"].min(), "–", _df_debug["SessionIndex"].max())

                unmatched_df = _df_debug[_df_debug["Session Time"].isna()][[
                    "Client Name", "Client: Insurance ID", "SessionIndex", "Date"
                ]].head(20)
                st.write("**Still-unmatched rows (first 20):**")
                st.dataframe(unmatched_df, width="stretch")

                matched_df = _df_debug[_df_debug["Session Time"].notna()][[
                    "Client Name", "Client: Insurance ID", "SessionIndex", "Date", "Session Time"
                ]].head(20)
                st.write("**Matched rows (first 20):**")
                st.dataframe(matched_df, width="stretch")
            # ---- END MERGE DEBUGGING ----

            _td = time.perf_counter()
            for src, dst in [
                ("Present_Client", "_Note_ClientPresent"),
                ("Present_ParentCaregiver", "_Note_ParentPresent"),
                ("Present_Sibling", "_Note_SiblingPresent"),
                ("Present_BT_RBT", "_Note_BTPresent"),
            ]:
                if src in df_f.columns:
                    df_f[dst] = df_f[src].fillna(False).astype(bool)
                else:
                    df_f[dst] = np.nan

            if "Session Time" in df_f.columns:
                df_f["Session Time"] = df_f["Session Time"].apply(
                    lambda v: v if isinstance(v, str) and ("AM" in v or "PM" in v)
                    else normalize_session_time(v)
                )

                # ---- Resolve ambiguous AM/PM: Excel times first, signature fallback ----
                ambiguous_mask = df_f["Session Time"].astype(str).str.contains("⚠️", na=False)

                if ambiguous_mask.any():
                    def resolve_ampm(row):
                        time_str = str(row["Session Time"])
                        clean = time_str.replace("  ⚠️ AM/PM unknown", "").strip()

                        # Primary: use Excel start/end datetimes — ground truth
                        excel_start = row.get("_ExcelStart_dt")
                        excel_end   = row.get("_ExcelEnd_dt")
                        if pd.notna(excel_start) and pd.notna(excel_end):
                            m = re.match(
                                r"(\d{1,2}:\d{2})\s*AM\s*-\s*(\d{1,2}:\d{2})\s*AM",
                                clean, re.I,
                            )
                            if m:
                                start_ap = "PM" if excel_start.hour >= 12 else "AM"
                                end_ap   = "PM" if excel_end.hour   >= 12 else "AM"
                                return f"{m.group(1)} {start_ap} - {m.group(2)} {end_ap}"

                        # Fallback: use parent caregiver signature time
                        sig_dt = row.get("_ParentSig_dt")
                        if pd.notna(sig_dt):
                            if sig_dt.hour >= 12:
                                return clean.replace("AM", "PM")
                            return clean

                        # No resolution possible — keep ⚠️ for manual review
                        return time_str

                    df_f.loc[ambiguous_mask, "Session Time"] = df_f[ambiguous_mask].apply(
                        resolve_ampm, axis=1
                    )

            df_f["Has External Session"] = (
                df_f.get("Session Time", pd.Series([np.nan] * len(df_f))).notna()
                & (df_f.get("Session Time").astype(str).str.strip() != "")
            )

            df_f[["_ExtStart_dt", "_ExtEnd_dt"]] = df_f.apply(
                lambda r: pd.Series(parse_session_time_range(r.get("Session Time"), r.get("Date"))),
                axis=1,
            )

            has_ext_valid = (
                df_f["Has External Session"]
                & df_f["_ExtStart_dt"].notna()
                & df_f["_ExtEnd_dt"].notna()
            )

            # Note Minutes — derived from PDF session time (what the BT documented)
            df_f["Note Minutes"] = np.nan
            df_f.loc[has_ext_valid, "Note Minutes"] = (
                (df_f.loc[has_ext_valid, "_ExtEnd_dt"] - df_f.loc[has_ext_valid, "_ExtStart_dt"])
                .dt.total_seconds()
                / 60.0
            )

            # _End_dt — used for signature timing check.
            # Prefer Excel end time (ground truth); fall back to PDF-parsed end time.
            has_excel_end = df_f["_ExcelEnd_dt"].notna()
            df_f.loc[has_excel_end, "_End_dt"] = df_f.loc[has_excel_end, "_ExcelEnd_dt"]
            df_f.loc[~has_excel_end & has_ext_valid, "_End_dt"] = (
                df_f.loc[~has_excel_end & has_ext_valid, "_ExtEnd_dt"]
            )

            # Duration Match: True if within 1 minute
            df_f["Duration Match"] = (
                df_f["Actual Minutes"].notna()
                & df_f["Note Minutes"].notna()
                & ((df_f["Actual Minutes"] - df_f["Note Minutes"]).abs() <= 1)
            )

            # ---- PDF Time vs Excel Time accuracy check ----
            def _pdf_time_matches_excel(row):
                pdf_time = str(row.get("Session Time", "") or "")
                if not pdf_time or pdf_time == "nan" or "⚠️" in pdf_time:
                    return np.nan  # still unresolved or missing — can't verify

                excel_start = row.get("_ExcelStart_dt")
                excel_end   = row.get("_ExcelEnd_dt")
                if pd.isna(excel_start) or pd.isna(excel_end):
                    return np.nan  # no Excel time to compare against

                m = re.match(
                    r"(\d{1,2}):(\d{2})\s*(AM|PM)\s*-\s*(\d{1,2}):(\d{2})\s*(AM|PM)",
                    pdf_time, re.I,
                )
                if not m:
                    return False

                def to_24h(h, ap):
                    return int(h) % 12 + (12 if ap.upper() == "PM" else 0)

                return (
                    to_24h(m.group(1), m.group(3)) == excel_start.hour
                    and int(m.group(2))             == excel_start.minute
                    and to_24h(m.group(4), m.group(6)) == excel_end.hour
                    and int(m.group(5))             == excel_end.minute
                )

            df_f["_PDF_Time_Accurate"] = df_f.apply(_pdf_time_matches_excel, axis=1)

            _log_step("tab2.merge_external_sessions_derived_times_and_checks", _td)

        else:
            st.warning("External sessions data provided, but required columns are missing.")
            _LOG.info(
                "tab2.merge_external_sessions: skipped (external data missing Insurance ID or Session Time)"
            )

    else:
        _LOG.info(
            "tab2.merge_external_sessions: skipped (no external_sessions_df in session state)"
        )

    # ---- BT contacts ----
    df_f["Phone"] = ""
    df_f["Email"] = ""

    if bt_contacts_file is not None:
        _t = time.perf_counter()
        _bt_blob = bt_contacts_file.getvalue()
        _staff_tuple = tuple(
            str(s) for s in df_f["Staff Name"].dropna().unique()
        )
        _phone_pairs, _email_pairs, _bt_err = _cached_bt_contact_maps(
            _bt_blob,
            bt_contacts_file.name or "contacts.csv",
            _staff_tuple,
        )
        if _bt_err is not None:
            st.error(_bt_err)
        else:
            _pmap = dict(_phone_pairs)
            _emap = dict(_email_pairs)
            df_f["Phone"] = _map_staff_to_contact_lookup(df_f["Staff Name"], _pmap)
            df_f["Email"] = _map_staff_to_contact_lookup(df_f["Staff Name"], _emap)
        _log_step("tab2.bt_contacts_lookup", _t)
    else:
        _LOG.info("tab2.bt_contacts_lookup: skipped (no BT contacts file)")

    # ---- Zoho Case Coordinator merge ----
    df_f["Case Coordinator Name"] = ""

    if zoho_file is not None:
        _t = time.perf_counter()
        zoho_df = read_any(zoho_file)
        if zoho_df is not None:
            zoho_df = normalize_cols(zoho_df)

            zoho_required = {"Medicaid ID", "Case Coordinator Name"}
            zoho_missing = zoho_required - set(zoho_df.columns)
            if zoho_missing:
                st.error(f"Zoho file is missing columns: {sorted(zoho_missing)}")
            else:
                # Normalize Medicaid ID for matching
                zoho_df["_medicaid_norm"] = zoho_df["Medicaid ID"].astype(str).str.strip().str.upper()

                # Parse DOB if available for fallback matching
                if "Date of Birth" in zoho_df.columns:
                    zoho_df["_dob"] = pd.to_datetime(zoho_df["Date of Birth"], errors="coerce")
                else:
                    zoho_df["_dob"] = pd.NaT

                # Build lookup dicts
                medicaid_to_cc = (
                    zoho_df.dropna(subset=["Medicaid ID"])
                    .set_index("_medicaid_norm")["Case Coordinator Name"]
                    .to_dict()
                )

                dob_to_cc = {}
                if zoho_df["_dob"].notna().any():
                    dob_to_cc = (
                        zoho_df.dropna(subset=["_dob"])
                        .drop_duplicates(subset=["_dob"])
                        .set_index("_dob")["Case Coordinator Name"]
                        .to_dict()
                    )

                # Primary match: Insurance ID → Medicaid ID
                insurance_id_col = "Client: Insurance ID"
                if insurance_id_col in df_f.columns:
                    df_f["_insured_norm"] = df_f[insurance_id_col].astype(str).str.strip().str.upper()
                    df_f["Case Coordinator Name"] = df_f["_insured_norm"].map(medicaid_to_cc).fillna("")
                    df_f.drop(columns=["_insured_norm"], inplace=True)

                # Fallback match: DOB
                if dob_to_cc and "Client: Date of Birth" in df_f.columns:
                    df_f["_dob_dt"] = pd.to_datetime(df_f["Client: Date of Birth"], errors="coerce")
                    missing_cc = df_f["Case Coordinator Name"] == ""
                    df_f.loc[missing_cc, "Case Coordinator Name"] = (
                        df_f.loc[missing_cc, "_dob_dt"].map(dob_to_cc).fillna("")
                    )
                    df_f.drop(columns=["_dob_dt"], inplace=True)

                matched_count = (df_f["Case Coordinator Name"] != "").sum()
                st.success(f"Zoho match: {matched_count} of {len(df_f)} sessions matched a Case Coordinator.")

        _log_step("tab2.zoho_case_coordinator_lookup", _t)
    else:
        _LOG.info("tab2.zoho_case_coordinator_lookup: skipped (no Zoho file)")

    _t = time.perf_counter()
    df_f["Daily Minutes"] = df_f.groupby(["Staff Name", "Date"])["Actual Minutes"].transform("sum")
    df_f["Daily Session Count"] = df_f.groupby(["Client Name", "Date", "Session"])["Session"].transform("count")

    # ---- Gap tracking columns (kept for data, check disabled) ----
    df_f["_SessionGapOk"] = True
    df_f["_SessionGapMinutes"] = np.nan

    if "_ExtStart_dt" in df_f.columns and "_ExtEnd_dt" in df_f.columns:
        for (client, date), grp in df_f.groupby(["Client Name", "Date"], sort=False):
            if len(grp) < 2:
                continue
            grp_sorted = grp.dropna(subset=["_ExtStart_dt", "_ExtEnd_dt"]).sort_values("_ExtStart_dt")
            if len(grp_sorted) < 2:
                continue
            idx_list = list(grp_sorted.index)
            for i in range(len(idx_list) - 1):
                a_end = df_f.at[idx_list[i], "_ExtEnd_dt"]
                b_start = df_f.at[idx_list[i + 1], "_ExtStart_dt"]
                gap_minutes = (b_start - a_end).total_seconds() / 60.0
                for idx in [idx_list[i], idx_list[i + 1]]:
                    existing = df_f.at[idx, "_SessionGapMinutes"]
                    if pd.isna(existing) or gap_minutes < existing:
                        df_f.at[idx, "_SessionGapMinutes"] = gap_minutes
                    if gap_minutes < MIN_SESSION_GAP_MINUTES:
                        df_f.at[idx, "_SessionGapOk"] = False
    _log_step("tab2.daily_aggregates_and_gap_tracking", _t)

    # Expose _PDF_Time_Accurate as a readable display column
    if "_PDF_Time_Accurate" in df_f.columns:
        df_f["PDF Time Matches Excel"] = df_f["_PDF_Time_Accurate"].map(
            {True: "✅ Match", False: "❌ Mismatch", np.nan: "—"}
        ).fillna("—")

    _t = time.perf_counter()
    eval_results = df_f.apply(evaluate_row, axis=1, result_type="expand")
    df_f["_DurationOk"] = eval_results["duration_ok"]
    df_f["_SigOk"] = eval_results["sig_ok"]
    df_f["_ExtOk"] = eval_results["ext_ok"]
    df_f["_DailyOk"] = eval_results["daily_ok"]
    df_f["_HasTimeAdjSig"] = eval_results["has_time_adj_sig"]
    df_f["_NoteOk"] = eval_results["note_ok"]
    df_f["_NoteParseOk"] = eval_results["note_parse_ok"]
    df_f["_PdfTimeOk"] = eval_results["pdf_time_ok"]
    df_f["_GapOk"] = eval_results["gap_ok"]
    df_f["_OverallPass"] = eval_results["overall_pass"]
    _log_step("tab2.validators_evaluate_row", _t)

    _t = time.perf_counter()
    df_f["Failure Reasons"] = df_f.apply(get_failure_reasons, axis=1)
    _log_step("tab2.failure_reasons_apply", _t)

    _t = time.perf_counter()
    for col in ["Adult Caregiver signature time"]:
        if col in df_f.columns:
            df_f[col] = (
                pd.to_datetime(df_f[col], errors="coerce", utc=True)
                .dt.tz_convert(None)
                .dt.strftime("%m/%d/%Y %I:%M:%S %p")
                .fillna("")
            )
    _log_step("tab2.format_signature_columns", _t)

    display_cols = [
        "AlohaABA Appointment ID",
        "Date",
        "Client Name",
        "Staff Name",
        "Phone",
        "Email",
        "Case Coordinator Name",
        "Duration",
        "Actual Minutes",
        "Notes Minutes",
        "Daily Minutes",
        "Daily Session Count",
        "Duration Match",
        "Adult Caregiver signature time",
        "Session Time",
        "PDF Time Matches Excel",
        "Note Compliance Errors",
    ]
    if TIME_ADJ_COL in df_f.columns:
        display_cols.append(TIME_ADJ_COL)

    display_cols.append("Failure Reasons")

    present_cols = [c for c in display_cols if c in df_f.columns]

    # Filter demo/marry FIRST before any display or export
    if "Client Name" in df_f.columns:
        demo_mask = df_f["Client Name"].astype(str).str.lower().str.contains("demo|marry", na=False)
        export_df = df_f[~demo_mask].copy()
    else:
        export_df = df_f.copy()

    st.subheader("2) Results")
    with st.expander("Summary", expanded=False):
        st.dataframe(export_df[present_cols], width="stretch", height=560)

    st.caption("Summary")
    summary = pd.DataFrame(
        {
            "Total (after filters)": [len(export_df)],
            "Pass": [int(export_df["_OverallPass"].sum())],
            "Fail": [int((~export_df["_OverallPass"]).sum())],
        }
    )
    st.table(summary)

    _t = time.perf_counter()
    st.session_state["session_checker_df"] = export_df.copy()
    st.session_state["session_checker_present_cols"] = present_cols
    _LOG.info(
        "tab2.checkpoint: session_checker_df stored — %d rows (Aloha file can be uploaded)",
        len(export_df),
    )
    _log_step("tab2.persist_session_state_for_billed_checker", _t)

    _t = time.perf_counter()
    xlsx_all = export_excel(export_df[present_cols])
    xlsx_clean = export_excel(export_df[export_df["_OverallPass"]][present_cols])
    xlsx_flagged = export_excel(export_df[~export_df["_OverallPass"]][present_cols])
    _log_step("tab2.build_three_download_xlsx_buffers", _t)

    _LOG.info(
        "tab2.done: full Session Checker rerun — total_wall=%.4f s — rows exported=%d",
        time.perf_counter() - _tab2_pipeline_start,
        len(export_df),
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Download All", data=xlsx_all, file_name="all_sessions.xlsx")
    with c2:
        st.download_button("✅ Passed Only", data=xlsx_clean, file_name="clean_sessions.xlsx")
    with c3:
        st.download_button("⬇️ Failed Only", data=xlsx_flagged, file_name="flagged_sessions.xlsx")


# =========================================================
# TAB 3: BILLED CHECKER
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
            "(must include 'Appointment ID', 'Date Billed', and 'Completed')",
            type=["xlsx", "xls", "csv"],
            key="billing_status_file",
        )

        if billing_file is not None:
            _t = time.perf_counter()
            try:
                billing_df = read_any(billing_file)
            except Exception as e:
                billing_df = None
                st.error(f"Error reading billing status file: {e}")
                _LOG.info("tab3.read_billing_upload: failed (%s)", e)

            if billing_df is not None:
                billing_df = normalize_cols(billing_df)
                _log_step("tab3.read_and_normalize_billing_file", _t)

                if not all(c in billing_df.columns for c in ["Appointment ID", "Date Billed", "Completed"]):
                    st.error("Billing status file must contain columns: 'Appointment ID', 'Date Billed', and 'Completed'.")
                elif "AlohaABA Appointment ID" not in base_df.columns:
                    st.error(
                        "Session Checker data is missing 'AlohaABA Appointment ID'. "
                        "Please ensure the HiRasmus export includes this column."
                    )
                else:
                    _t = time.perf_counter()
                    merged = base_df.merge(
                        billing_df[["Appointment ID", "Date Billed", "Completed"]],
                        left_on="AlohaABA Appointment ID",
                        right_on="Appointment ID",
                        how="left",
                        sort=False,
                    )

                    demo_mask = merged["Client Name"].astype(str).str.lower().str.contains("demo|marry", na=False)
                    merged = merged[~demo_mask].copy()

                    def classify_status(row):
                        app_id = str(row.get("Appointment ID", "")).strip()
                        date_billed = str(row.get("Date Billed", "")).strip()

                        if app_id == "" or app_id.lower() == "nan":
                            return "No Match"
                        if date_billed != "" and date_billed.lower() != "nan":
                            return "Billed"
                        return "Unbilled"

                    merged["Billing Status"] = merged.apply(classify_status, axis=1)
                    _log_step("tab3.merge_aloha_billing_and_classify_status", _t)

                    # Flag passed sessions where Aloha shows not completed
                    not_completed_mask = (
                        merged["_OverallPass"] == True
                    ) & (
                        merged["Billing Status"] != "No Match"
                    ) & (
                        merged["Completed"].astype(str).str.strip().str.lower() != "yes"
                    )

                    merged.loc[not_completed_mask, "Failure Reasons"] = (
                        merged.loc[not_completed_mask, "Failure Reasons"]
                        .apply(lambda x: "Authorization Exceeded" if x == "PASS" else x + "; Authorization Exceeded")
                    )
                    merged.loc[not_completed_mask, "_OverallPass"] = False

                    all_df = merged.copy()
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
- **No Match (no Appointment ID match):** {nomatch_count}
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
                        for c in dict.fromkeys(
                            present_cols + ["Appointment ID", "Date Billed", "Billing Status"]
                        ).keys()
                        if c in merged.columns
                    ]

                    _t = time.perf_counter()
                    xlsx_all = export_excel(all_df[dl_cols])
                    xlsx_billed = export_excel(billed_df[dl_cols])
                    xlsx_unbilled_clean = export_excel(unbilled_clean_df[dl_cols])
                    xlsx_unbilled_flagged = export_excel(unbilled_flagged_df[dl_cols])
                    xlsx_nomatch = export_excel(nomatch_df[dl_cols])
                    _log_step("tab3.build_five_download_xlsx_buffers", _t)
                    _LOG.info(
                        "tab3.done: billed checker complete — %d sessions in merged view",
                        total_sessions,
                    )

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