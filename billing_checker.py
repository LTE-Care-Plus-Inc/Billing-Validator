# ===========================
# app.py
# ===========================
import re
from difflib import SequenceMatcher

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
            text = pdf_bytes_to_text(pdf_file.read(), preserve_layout=True)
            results = parse_notes(text)

            if not results:
                st.warning("No notes found in the PDF. Make sure it contains 'Client:' blocks.")
            else:
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
                    st.dataframe(notes_df[preview_cols].head(120), use_container_width=True)

                cols = [
                    "Client",
                    "Session Time",
                    "Session Date",
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

                st.info(
                    f"Generated {len(ext_df)} external session rows. "
                    "These will be used automatically in the **Session Checker** tab."
                )

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
            external_sessions_df = read_any(manual_external_file)

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

    df = pd.read_excel(sessions_file, dtype=object)
    df = normalize_cols(df)

    if "Start date" not in df.columns and "Start Date" in df.columns:
        df = df.rename(columns={"Start Date": "Start date"})

    if not ensure_cols(df, REQ_COLS, "Sessions File"):
        st.stop()

    if USE_TIME_ADJ_OVERRIDE:
        if not ensure_cols(df, [TIME_ADJ_COL], "Sessions File"):
            st.stop()

    df_f = df[
        (df["Status"].astype(str).str.strip() == STATUS_REQUIRED)
        & (df["Session"].astype(str).str.strip() == SESSION_REQUIRED)
    ].copy()

    df_f["_RowOrder"] = np.arange(len(df_f))
    df_f["Actual Minutes"] = df_f["Duration"].apply(parse_duration_to_minutes)

    start_raw = df_f["Start date"].astype(str).str.strip()
    start_clean = start_raw.str.split().str[0]
    df_f["Date"] = pd.to_datetime(start_clean, errors="coerce").dt.strftime("%m/%d/%Y")

    df_f["_End_dt"] = pd.NaT
    df_f["_ParentSig_dt"] = pd.to_datetime(df_f["Adult Caregiver signature time"], errors="coerce")

    df_f["Staff Name"] = df_f["User"].apply(normalize_client_name_for_match)
    df_f["Client Name"] = df_f["Client"].apply(normalize_client_name_for_match)

    # ---- External session matching ----
    if external_sessions_df is not None:
        ext_df = normalize_cols(external_sessions_df.copy())

        if ensure_cols(ext_df, ["Client", "Session Time"], "External Sessions File"):
            ext_df["Client Name"] = ext_df["Client"].apply(normalize_client_name_for_match)

            ext_df["SessionIndex"] = ext_df.groupby("Client Name").cumcount()
            df_f["SessionIndex"] = df_f.groupby("Client Name").cumcount()

            merge_cols = [
                "Client Name",
                "SessionIndex",
                "Session Time",
                "Session Date",
                "Present_Client",
                "Present_ParentCaregiver",
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

            df_f = df_f.sort_values("_RowOrder", kind="mergesort").reset_index(drop=True)

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

            df_f.loc[has_ext_valid, "Actual Minutes"] = (
                (df_f.loc[has_ext_valid, "_ExtEnd_dt"] - df_f.loc[has_ext_valid, "_ExtStart_dt"])
                .dt.total_seconds()
                / 60.0
            )
            df_f.loc[has_ext_valid, "_End_dt"] = df_f.loc[has_ext_valid, "_ExtEnd_dt"]

        else:
            st.warning("External sessions data provided, but required columns are missing.")

    # ---- BT contacts ----
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
                        staff_to_phone[staff] = best_row.get("Phone", "")
                        staff_to_email[staff] = best_row.get("Email", "")

                df_f["Phone"] = df_f["Staff Name"].map(staff_to_phone).fillna("")
                df_f["Email"] = df_f["Staff Name"].map(staff_to_email).fillna("")

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

    eval_results = df_f.apply(evaluate_row, axis=1, result_type="expand")
    df_f["_DurationOk"] = eval_results["duration_ok"]
    df_f["_SigOk"] = eval_results["sig_ok"]
    df_f["_ExtOk"] = eval_results["ext_ok"]
    df_f["_DailyOk"] = eval_results["daily_ok"]
    df_f["_HasTimeAdjSig"] = eval_results["has_time_adj_sig"]
    df_f["_NoteOk"] = eval_results["note_ok"]
    df_f["_NoteParseOk"] = eval_results["note_parse_ok"]
    df_f["_GapOk"] = eval_results["gap_ok"]
    df_f["_OverallPass"] = eval_results["overall_pass"]

    df_f["Failure Reasons"] = df_f.apply(get_failure_reasons, axis=1)

    for col in ["Adult Caregiver signature time"]:
        if col in df_f.columns:
            df_f[col] = (
                pd.to_datetime(df_f[col], errors="coerce")
                .dt.strftime("%m/%d/%Y %I:%M:%S %p")
                .fillna("")
            )

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
        "Daily Session Count",
        "Adult Caregiver signature time",
        "Session Time",
        "Note Compliance Errors",
    ]
    if TIME_ADJ_COL in df_f.columns:
        display_cols.append(TIME_ADJ_COL)

    display_cols.append("Failure Reasons")

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
            try:
                billing_df = read_any(billing_file)
            except Exception as e:
                billing_df = None
                st.error(f"Error reading billing status file: {e}")

            if billing_df is not None:
                billing_df = normalize_cols(billing_df)

                if not all(c in billing_df.columns for c in ["Appointment ID", "Date Billed", "Completed"]):
                    st.error("Billing status file must contain columns: 'Appointment ID', 'Date Billed', and 'Completed'.")
                elif "AlohaABA Appointment ID" not in base_df.columns:
                    st.error(
                        "Session Checker data is missing 'AlohaABA Appointment ID'. "
                        "Please ensure the HiRasmus export includes this column."
                    )
                else:
                    merged = base_df.merge(
                        billing_df[["Appointment ID", "Date Billed", "Completed"]],
                        left_on="AlohaABA Appointment ID",
                        right_on="Appointment ID",
                        how="left",
                        sort=False,
                    )

                    def classify_status(row):
                        app_id = str(row.get("Appointment ID", "")).strip()
                        date_billed = str(row.get("Date Billed", "")).strip()

                        if app_id == "" or app_id.lower() == "nan":
                            return "No Match"
                        if date_billed != "" and date_billed.lower() != "nan":
                            return "Billed"
                        return "Unbilled"

                    merged["Billing Status"] = merged.apply(classify_status, axis=1)

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

                    xlsx_all = export_excel(all_df[dl_cols])
                    xlsx_billed = export_excel(billed_df[dl_cols])
                    xlsx_unbilled_clean = export_excel(unbilled_clean_df[dl_cols])
                    xlsx_unbilled_flagged = export_excel(unbilled_flagged_df[dl_cols])
                    xlsx_nomatch = export_excel(nomatch_df[dl_cols])

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