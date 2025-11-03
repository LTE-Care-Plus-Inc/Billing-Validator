# app.py — HiRasmus (actual) vs Aloha (billing) checker
# Changes in this version:
# - Removed "Activity type" from required File 1 columns (still shown if present).
# - Output shows only one Appointment ID column: "Appointment ID" (from File 2).
# - All prior rules remain, incl. File 1 Status filter and Service Name filter on File 2.

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HiRasmus ↔ Aloha Duration & Billing Check", layout="wide")
st.title("HiRasmus ↔ Aloha — Duration & Billing Alignment (Service + Status filters)")

# ---------- Config ----------
REQ_FILE1 = [
    "Status",
    "Client",
    "Start time",
    "End time",
    "Duration",
    "AlohaABA Appointment ID",
    "Location (start)",
    "Location (end)",
    "User signature location",
    "Parent signature location",
]

REQ_FILE2 = ["Appointment ID", "Billing Minutes", "Staff Name", "Client Name", "Appt. Date", "Service Name"]

SERVICE_REQUIRED = "Direct Service BT"
STATUS_REQUIRED  = "Transferred to AlohaABA"

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
    """Convert duration 'HH:MM:SS' → minutes (float)."""
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

def minutes_to_hhmmss_signed(mins: float) -> str:
    """Format signed minutes → ±HH:MM:SS (negative for under)."""
    if pd.isna(mins):
        return ""
    sign = "-" if mins < 0 else ""
    secs_total = int(round(abs(mins) * 60))
    h = secs_total // 3600
    rem = secs_total % 3600
    m = rem // 60
    s = rem % 60
    return f"{sign}{int(h):02d}:{int(m):02d}:{int(s):02d}"

def export_excel(df: pd.DataFrame) -> bytes:
    """Export one sheet per Appt. Date."""
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

# ---------- Prefilters ----------
# File 2: Service filter
df2_f = df2[df2["Service Name"].astype(str).str.strip() == SERVICE_REQUIRED].copy()

# File 1: Status filter
df1_f = df1[df1["Status"].astype(str).str.strip() == STATUS_REQUIRED].copy()

st.caption(
    f'Prefilters → File 2: Service Name == "{SERVICE_REQUIRED}" | '
    f'File 1: Status == "{STATUS_REQUIRED}"'
)
st.write({
    "File 1 rows (before/after)": f"{int(df1.shape[0])} / {int(df1_f.shape[0])}",
    "File 2 rows (before/after)": f"{int(df2.shape[0])} / {int(df2_f.shape[0])}",
})

# ---------- Parse Duration ----------
df1_f["Actual Minutes"] = df1_f["Duration"].apply(parse_duration_to_minutes)

# ---------- Merge ----------
merged = pd.merge(
    df1_f,
    df2_f[["Appointment ID", "Billing Minutes", "Staff Name", "Client Name", "Appt. Date", "Service Name"]],
    left_on="AlohaABA Appointment ID",
    right_on="Appointment ID",
    how="left"
)

# ---------- Convert Billing ----------
merged["Scheduled Minutes"] = pd.to_numeric(merged["Billing Minutes"], errors="coerce")

# ---------- Checks (internal only; not in output) ----------
# 1) Duration window
check_duration = merged["Actual Minutes"].apply(lambda m: (not pd.isna(m)) and (m > MIN_MINUTES) and (m < MAX_MINUTES))

# 2) BOTH locations present
loc_start_ok = merged["Location (start)"].notna() & (merged["Location (start)"].astype(str).str.strip() != "")
loc_end_ok   = merged["Location (end)"].notna() & (merged["Location (end)"].astype(str).str.strip() != "")
check_locations = loc_start_ok & loc_end_ok

# 3) Billing alignment ±8 min
def bill_align(actual, billing):
    if pd.isna(actual) or pd.isna(billing):
        return False
    return abs(actual - billing) <= BILLING_TOL_MIN

check_billing = merged.apply(lambda r: bill_align(r["Actual Minutes"], r["Scheduled Minutes"]), axis=1)

merged["_OverallPass"] = check_duration & check_locations & check_billing

# ---------- Computed display fields ----------
merged["Δ vs Billing (minutes)"] = merged["Actual Minutes"] - merged["Scheduled Minutes"]
merged["Δ vs Billing (HH:MM:SS)"] = merged["Δ vs Billing (minutes)"].apply(minutes_to_hhmmss_signed)

# ---------- Display (only one Appointment ID; hide staff profile name) ----------
display_cols = [
    "Status",
    "Appt. Date",
    "Appointment ID",
    "Client",
    "Staff Name",
    "Start time",
    "End time",
    "Duration",
    "Location (start)",
    "Location (end)",
    "User signature location",
    "Parent signature location",
    "Service Name",
    "Scheduled Minutes",
    "Actual Minutes",
    "Δ vs Billing (HH:MM:SS)",
]
present_cols = [c for c in display_cols if c in merged.columns]

st.subheader("2) Results")
st.dataframe(merged[present_cols], use_container_width=True, height=560)

# ---------- Summary ----------
st.caption("Summary")
summary = pd.DataFrame({
    "Total (after filters)": [len(merged)],
    "Pass": [int(merged["_OverallPass"].sum())],
    "Fail": [int((~merged["_OverallPass"]).sum())],
})
st.table(summary)

# ---------- Export (no boolean columns in files) ----------
all_df = merged[present_cols].copy()
clean_df = merged[merged["_OverallPass"]][present_cols].copy()
flagged_df = merged[~merged["_OverallPass"]][present_cols].copy()

xlsx_all = export_excel(all_df)
xlsx_clean = export_excel(clean_df)
xlsx_flagged = export_excel(flagged_df)

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("⬇️ Download All by Date", data=xlsx_all, file_name="all_by_date.xlsx")
with c2:
    st.download_button("✅ Ready to Bill", data=xlsx_clean, file_name="clean_by_date.xlsx")
with c3:
    st.download_button("⚠️ Flagged", data=xlsx_flagged, file_name="flagged_by_date.xlsx")

st.caption(
    f"""
**Internal checks (not shown in table):**
- File 2 filtered to **Service Name == "{SERVICE_REQUIRED}"**
- File 1 filtered to **Status == "{STATUS_REQUIRED}"**
- Duration strictly > {MIN_MINUTES} and < {MAX_MINUTES} minutes
- **Both** Location (start) and Location (end) must have content
- Billing within ±{BILLING_TOL_MIN} minutes (see signed Δ column)
"""
)
