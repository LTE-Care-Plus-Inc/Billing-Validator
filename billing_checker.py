# app.py — HiRasmus (actual) vs Aloha (billing) checker
# Spec (final, with Service Name filter):
# - File 1 (HiRasmus): Status, Duration, User, Session, AlohaABA Appointment ID, Location (start),
#                      Staff Profile: Full Name (as shown on ID), Client
# - File 2 (Aloha): Appointment ID, Billing Minutes, Staff Name, Client Name, Appt.Date, Service Name
#
# Filters & Checks:
#   (A) File 2 prefilter → Service Name == "Direct Service BT"
#   1) Duration in minutes strictly > 60 and < 360
#   2) Location (start) not blank
#   3) |DurationMinutes - BillingMinutes| <= 8
#
# Output: Excel with All / Clean / Flagged sheets (one sheet per Appt.Date)

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HiRasmus ↔ Aloha Duration & Billing Check", layout="wide")
st.title("HiRasmus ↔ Aloha — Duration & Billing Alignment (with Service Filter)")

# ---------- Config ----------
REQ_FILE1 = [
    "Status", "Duration", "User", "Session",
    "AlohaABA Appointment ID", "Location (start)",
    "Staff Profile: Full Name (as shown on ID)", "Client",
]
REQ_FILE2 = ["Appointment ID", "Billing Minutes", "Staff Name", "Client Name", "Appt. Date", "Service Name"]

SERVICE_REQUIRED = "Direct Service BT"
BILLING_TOL_MIN = 8
MIN_MINUTES = 60
MAX_MINUTES = 360

# ---------- Helpers ----------
def ensure_cols(df: pd.DataFrame, cols, label: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"**{label}** missing required columns: {missing}")
        return False
    return True


def parse_duration_to_minutes(d):
    """Convert duration string 'HH:MM:SS' → minutes (float)."""
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
        pass
    return np.nan


def export_excel(df: pd.DataFrame) -> bytes:
    """Export one sheet per Appt.Date."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        if "Appt. Date" in df.columns:
            df["_ApptDate_"] = pd.to_datetime(df["Appt. Date"], errors="coerce").dt.date.astype("string")
            df["_ApptDate_"] = df["_ApptDate_"].fillna("Unknown")
            for date_key, sub in df.groupby("_ApptDate_"):
                sheet = str(date_key)[:31] if date_key else "Unknown"
                sub.drop(columns=["_ApptDate_"], errors="ignore").to_excel(w, index=False, sheet_name=sheet)
        else:
            df.to_excel(w, index=False, sheet_name="All")
    return buf.getvalue()


# ---------- UI ----------
st.subheader("1) Upload Files")
c1, c2 = st.columns(2)
with c1:
    f1 = st.file_uploader("File 1 — HiRasmus (Excel)", type=["xlsx", "xls"])
with c2:
    f2 = st.file_uploader("File 2 — Aloha Billing (Excel)", type=["xlsx", "xls"])

if not (f1 and f2):
    st.info("Upload both files to continue.")
    st.stop()

df1 = pd.read_excel(f1, dtype=object)
df2 = pd.read_excel(f2, dtype=object)

if not (ensure_cols(df1, REQ_FILE1, "File 1 (HiRasmus)") and ensure_cols(df2, REQ_FILE2, "File 2 (Aloha)")):
    st.stop()

# ---------- Clean up ----------
for df in (df1, df2):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

# ---------- Prefilter File 2 ----------
df2_f = df2[df2["Service Name"].astype(str).str.strip() == SERVICE_REQUIRED].copy()
st.caption(f'Prefilter applied to **File 2**: Service Name == "{SERVICE_REQUIRED}"')
st.write({
    "File 1 rows": int(df1.shape[0]),
    "File 2 rows before filter": int(df2.shape[0]),
    "File 2 rows after filter": int(df2_f.shape[0]),
})

# ---------- Parse Duration ----------
df1["_ActualMinutes"] = df1["Duration"].apply(parse_duration_to_minutes)

# ---------- Merge ----------
merged = pd.merge(
    df1,
    df2_f[["Appointment ID", "Billing Minutes", "Staff Name", "Client Name", "Appt. Date", "Service Name"]],
    left_on="AlohaABA Appointment ID",
    right_on="Appointment ID",
    how="left"
)

# ---------- Convert Billing ----------
merged["_BillingMinutes"] = pd.to_numeric(merged["Billing Minutes"], errors="coerce")

# ---------- Checks ----------
# 1) Duration window
merged["Check_DurationWindow"] = merged["_ActualMinutes"].apply(
    lambda m: (not pd.isna(m)) and (m > MIN_MINUTES) and (m < MAX_MINUTES)
)

# 2) Location present
merged["Check_LocationPresent"] = merged["Location (start)"].notna() & (
    merged["Location (start)"].astype(str).str.strip() != ""
)

# 3) Billing alignment ±8 min
def bill_align(actual, billing):
    if pd.isna(actual) or pd.isna(billing):
        return False
    return abs(actual - billing) <= BILLING_TOL_MIN

merged["Check_BillingAlign"] = merged.apply(
    lambda r: bill_align(r["_ActualMinutes"], r["_BillingMinutes"]), axis=1
)

# Overall Pass
merged["Overall Pass"] = (
    merged["Check_DurationWindow"]
    & merged["Check_LocationPresent"]
    & merged["Check_BillingAlign"]
)

# ---------- Display ----------
show_cols = [
    "AlohaABA Appointment ID", "Appt. Date", "Service Name", "Client", "Client Name",
    "Staff Profile: Full Name (as shown on ID)", "Staff Name",
    "Location (start)", "Duration", "_ActualMinutes",
    "_BillingMinutes", "Check_DurationWindow",
    "Check_LocationPresent", "Check_BillingAlign", "Overall Pass",
]
present_cols = [c for c in show_cols if c in merged.columns]

st.subheader("2) Results")
st.dataframe(merged[present_cols], use_container_width=True, height=520)

# ---------- Summary ----------
summary = pd.DataFrame({
    "Total": [len(merged)],
    "Pass": [int(merged["Overall Pass"].sum())],
    "Fail": [int((~merged["Overall Pass"]).sum())],
})
st.table(summary)

fail_reasons = pd.DataFrame({
    "Duration out of range": [int((~merged["Check_DurationWindow"]).sum())],
    "Location missing": [int((~merged["Check_LocationPresent"]).sum())],
    "Billing misaligned": [int((~merged["Check_BillingAlign"]).sum())],
})
st.table(fail_reasons)

# ---------- Export ----------
all_df = merged[present_cols].copy()
clean_df = merged[merged["Overall Pass"]][present_cols].copy()
flagged_df = merged[~merged["Overall Pass"]][present_cols].copy()

xlsx_all = export_excel(all_df)
xlsx_clean = export_excel(clean_df)
xlsx_flagged = export_excel(flagged_df)

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("⬇️ Download All by Date", data=xlsx_all, file_name="all_by_date.xlsx")
with c2:
    st.download_button("✅ Download Clean by Date", data=xlsx_clean, file_name="clean_by_date.xlsx")
with c3:
    st.download_button("⚠️ Download Flagged by Date", data=xlsx_flagged, file_name="flagged_by_date.xlsx")

st.caption(
    f"""
**Checks summary:**
- File 2 filtered to **Service Name == "{SERVICE_REQUIRED}"**  
- Duration strictly > {MIN_MINUTES} min and < {MAX_MINUTES} min  
- Location (start) must not be blank  
- Duration within ±{BILLING_TOL_MIN} minutes of Billing Minutes  
"""
)
