# app.py — HiRasmus (actual) vs Aloha (billing) checker

import io
import math
import json
import re
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HiRasmus ↔ Aloha Duration & Billing Check", layout="wide")
st.title("HiRasmus ↔ Aloha — Duration & Billing Alignment (Service + Status + Geo filters)")

# ---------- Config ----------
REQ_FILE1 = [
    "Status",
    "Client",
    "Start time",
    "End time",
    "Duration",
    "AlohaABA Appointment ID",
]

REQ_FILE2 = [
    "Appointment ID",
    "Billing Minutes",
    "Staff Name",
    "Client Name",
    "Appt. Date",
    "Service Name",
    "Appt. Start Time",
    "Appt. End Time",
    "Appointment Location",
]

MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "")

# ---------- Sidebar settings ----------
st.sidebar.header("Geo Settings")
GEO_TOL_FT = st.sidebar.number_input(
    "Geo distance tolerance (feet)",
    min_value=100,
    max_value=5000,
    value=800,   # default
    step=50,
)

SERVICE_REQUIRED = "Direct Service BT"
STATUS_REQUIRED = "Transferred to AlohaABA"

SIGN_TOL_MIN = 10         # ± minutes for signature vs Appt. End Time
BILLING_TOL_MIN = 8       # ± minutes for actual vs billed duration
MIN_MINUTES = 60
MAX_MINUTES = 360         # (currently not enforced)

if not MAPBOX_TOKEN:
    st.warning(
        "No MAPBOX_TOKEN found in Streamlit secrets. "
        "Geocoding Appointment Location will only work if it's already coordinates."
    )

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

# ---- Geo + time helpers ----
def meters_to_feet(m: float) -> float:
    return m / 0.3048

def haversine_meters(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    from math import radians, sin, cos, asin, sqrt
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlmb/2)**2
    return 2 * R * asin(sqrt(a))

def distance_ft(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return round(meters_to_feet(haversine_meters(p1[0], p1[1], p2[0], p2[1])), 1)

def parse_coord(value: Any) -> Optional[Tuple[float, float]]:
    """
    Accepts: 'lat, lon' / 'lon, lat' / 'POINT(lon lat)' / JSON with lat/lon or GeoJSON coordinates.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        a, b = float(value[0]), float(value[1])
        if -90 <= a <= 90 and -180 <= b <= 180:
            return (a, b)
        if -90 <= b <= 90 and -180 <= a <= 180:
            return (b, a)
        return None

    s = str(value).strip()

    # JSON/dict
    try:
        js = json.loads(s)
        if isinstance(js, dict):
            if "lat" in js and ("lon" in js or "lng" in js):
                return (float(js["lat"]), float(js.get("lon", js.get("lng"))))
            if "coordinates" in js and isinstance(js["coordinates"], (list, tuple)) and len(js["coordinates"]) == 2:
                lon, lat = js["coordinates"]
                return (float(lat), float(lon))  # GeoJSON [lon, lat]
    except Exception:
        pass

    # POINT(lon lat)
    m = re.match(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)", s, re.IGNORECASE)
    if m:
        lon, lat = float(m.group(1)), float(m.group(2))
        return (lat, lon)

    # "a, b"
    if "," in s:
        a, b = [t.strip() for t in s.split(",", 1)]
        try:
            a, b = float(a), float(b)
            if -90 <= a <= 90 and -180 <= b <= 180:
                return (a, b)
            if -90 <= b <= 90 and -180 <= a <= 180:
                return (b, a)
        except Exception:
            return None
    return None

def mapbox_forward_geocode(address: str, token: str) -> Optional[Tuple[float, float]]:
    if not address or not address.strip() or not token:
        return None

    import requests
    url = (
        "https://api.mapbox.com/geocoding/v5/mapbox.places/"
        + requests.utils.quote(address)
        + ".json"
    )
    params = {
        "access_token": token,
        "limit": 1,
        "country": "US",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        features = data.get("features") or []
        if not features:
            return None
        lon, lat = features[0]["geometry"]["coordinates"]
        return (float(lat), float(lon))
    except Exception:
        return None

def within_time_tol_ignoring_date(sig_ts, base_ts, tol_min):
    """Compare sig vs base in minutes, ignoring date (only time-of-day)."""
    if pd.isna(sig_ts) or pd.isna(base_ts):
        return False
    sig_ts = sig_ts.replace(year=base_ts.year, month=base_ts.month, day=base_ts.day)
    diff_min = abs((sig_ts - base_ts).total_seconds()) / 60.0
    return diff_min <= tol_min

# ---------- Geo checks: compare signature locations to Appointment Location ----------
def eval_geo(row):
    appt_loc_raw   = row.get("Appointment Location")
    parent_loc_raw = row.get("Parent signature location")
    user_loc_raw   = row.get("User signature location")

    # Default outputs
    parent_dist = None
    user_dist = None
    ok = False
    status = "Geo not checked"
    code = "NOT_CHECKED"
    reason = ""

    # 0) Appointment Location required to do geo
    if pd.isna(appt_loc_raw) or str(appt_loc_raw).strip() == "":
        status = "Geo not checked - no Appointment Location"
        code = "NO_APPT_LOC"
        reason = "Appointment Location is blank."
        return pd.Series({
            "Geo Status": status,
            "Geo Reason Code": code,
            "Geo Reason": reason,
            "Geo Parent↔Appt ft": parent_dist,
            "Geo User↔Appt ft": user_dist,
            "check_geo": ok,
        })

    # 1) Signature locations required
    if pd.isna(parent_loc_raw) or str(parent_loc_raw).strip() == "" or \
       pd.isna(user_loc_raw) or str(user_loc_raw).strip() == "":
        status = "Geo not checked - missing signature locations"
        code = "NO_SIG_LOC"
        reason = "Parent and/or User signature location is missing."
        return pd.Series({
            "Geo Status": status,
            "Geo Reason Code": code,
            "Geo Reason": reason,
            "Geo Parent↔Appt ft": parent_dist,
            "Geo User↔Appt ft": user_dist,
            "check_geo": ok,
        })

    # 2) Parse Parent/User coordinates
    p_parent = parse_coord(parent_loc_raw)
    p_user   = parse_coord(user_loc_raw)
    if not p_parent or not p_user:
        status = "Geo not checked - invalid signature coordinates"
        code = "BAD_SIG_COORD"
        reason = "Could not parse Parent/User signature coordinates."
        return pd.Series({
            "Geo Status": status,
            "Geo Reason Code": code,
            "Geo Reason": reason,
            "Geo Parent↔Appt ft": parent_dist,
            "Geo User↔Appt ft": user_dist,
            "check_geo": ok,
        })

    # 3) Appointment Location: try coords first, then Mapbox geocode the address
    p_appt = parse_coord(appt_loc_raw)
    if not p_appt:
        if not MAPBOX_TOKEN:
            status = "Geo not checked - no Mapbox token"
            code = "NO_TOKEN"
            reason = "MAPBOX_TOKEN is missing; cannot geocode Appointment Location."
            return pd.Series({
                "Geo Status": status,
                "Geo Reason Code": code,
                "Geo Reason": reason,
                "Geo Parent↔Appt ft": parent_dist,
                "Geo User↔Appt ft": user_dist,
                "check_geo": ok,
            })
        p_appt = mapbox_forward_geocode(str(appt_loc_raw), MAPBOX_TOKEN)
        if not p_appt:
            status = "Geo not checked - bad Appointment Location address"
            code = "BAD_ADDR"
            reason = "Mapbox could not geocode the Appointment Location address."
            return pd.Series({
                "Geo Status": status,
                "Geo Reason Code": code,
                "Geo Reason": reason,
                "Geo Parent↔Appt ft": parent_dist,
                "Geo User↔Appt ft": user_dist,
                "check_geo": ok,
            })

    # 4) We have all three coordinate sets: compute distances
    parent_dist = distance_ft(p_appt, p_parent)
    user_dist   = distance_ft(p_appt, p_user)

    if parent_dist <= GEO_TOL_FT and user_dist <= GEO_TOL_FT:
        ok = True
        status = "Within geo range"
        code = "WITHIN"
        reason = (
            f"Parent and User sig locations are within {GEO_TOL_FT} ft of Appointment Location "
            f"(Parent={parent_dist} ft, User={user_dist} ft)."
        )
    else:
        ok = False
        status = "Outside geo range"
        code = "OUTSIDE"
        reason = (
            f"At least one signature location is more than {GEO_TOL_FT} ft from Appointment Location "
            f"(Parent={parent_dist} ft, User={user_dist} ft)."
        )

    return pd.Series({
        "Geo Status": status,
        "Geo Reason Code": code,
        "Geo Reason": reason,
        "Geo Parent↔Appt ft": parent_dist,
        "Geo User↔Appt ft": user_dist,
        "check_geo": ok,
    })


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

# ---------- Let user pick location columns (only location fields) ----------
possible_location_cols = [
    "Location (start)",
    "Location (end)",
    "User signature location",
    "Parent signature location",
]

available_location_cols = [c for c in possible_location_cols if c in df1.columns]

if not available_location_cols:
    available_location_cols = [c for c in df1.columns if "location" in c.lower()]

st.markdown("**Select the location fields to validate (1–4):**")

selected_location_cols = []
num_cols = 4
cols = st.columns(num_cols)

for i, col_name in enumerate(available_location_cols):
    with cols[i % num_cols]:
        default_checked = col_name in ["Location (start)", "Location (end)"]
        checked = st.checkbox(col_name, value=default_checked, key=f"loc_{col_name}")
        if checked:
            selected_location_cols.append(col_name)

# enforce max 4
if len(selected_location_cols) == 0:
    st.warning("No location fields selected — location check will fail.")
elif len(selected_location_cols) > 4:
    st.warning("You selected more than 4 location fields — only the first 4 will be used.")
    selected_location_cols = selected_location_cols[:4]

# ---------- Clean up ----------
for df in (df1, df2):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

# ---------- Prefilters ----------
df2_f = df2[df2["Service Name"].astype(str).str.strip() == SERVICE_REQUIRED].copy()
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
    df2_f[
        [
            "Appointment ID",
            "Billing Minutes",
            "Staff Name",
            "Client Name",
            "Appt. Date",
            "Service Name",
            "Appt. Start Time",
            "Appt. End Time",
            "Appointment Location",
        ]
    ],
    left_on="AlohaABA Appointment ID",
    right_on="Appointment ID",
    how="left",
)

# ---------- Normalize datetime columns for signature checks ----------
merged["_ApptEnd_dt"] = pd.to_datetime(merged["Appt. End Time"], errors="coerce")

if "Parent signature time" in merged.columns:
    merged["_ParentSig_dt"] = pd.to_datetime(merged["Parent signature time"], errors="coerce")
else:
    merged["_ParentSig_dt"] = pd.NaT

if "User signature time" in merged.columns:
    merged["_UserSig_dt"] = pd.to_datetime(merged["User signature time"], errors="coerce")
else:
    merged["_UserSig_dt"] = pd.NaT

# ---------- Convert Billing ----------
merged["Scheduled Minutes"] = pd.to_numeric(merged["Billing Minutes"], errors="coerce")

# ---------- Geo with progress bar ----------
st.subheader("2) Geolocation Check (Appointment Location vs Signatures)")
geo_progress = st.progress(0.0)
geo_rows = []
total_rows = len(merged) if len(merged) > 0 else 1

for i, (_, r) in enumerate(merged.iterrows()):
    geo_rows.append(eval_geo(r))
    geo_progress.progress((i + 1) / total_rows)

geo_res = pd.DataFrame(geo_rows)
merged = pd.concat([merged.reset_index(drop=True), geo_res.reset_index(drop=True)], axis=1)
check_geo = merged["check_geo"].fillna(False)

# ---------- Checks ----------
# 1) Duration window
check_duration = merged["Actual Minutes"].apply(lambda m: (not pd.isna(m)) and (m >= MIN_MINUTES))

# 2) BOTH locations present (selected)
if selected_location_cols:
    import operator
    from functools import reduce

    loc_checks = []
    for col in selected_location_cols:
        has_val = merged[col].notna() & (merged[col].astype(str).str.strip() != "")
        loc_checks.append(has_val)
    check_locations = reduce(operator.and_, loc_checks)
else:
    check_locations = pd.Series(False, index=merged.index)

# 3) Billing alignment ±BILLING_TOL_MIN min
def bill_align(actual, billing):
    if pd.isna(actual) or pd.isna(billing):
        return False
    return abs(actual - billing) < BILLING_TOL_MIN

check_billing = merged.apply(
    lambda r: bill_align(r["Actual Minutes"], r["Scheduled Minutes"]),
    axis=1
)

# 4) Signature-time alignment ±SIGN_TOL_MIN min vs Appt. End Time
def sig_times_ok(row):
    base_ts = row.get("_ApptEnd_dt", pd.NaT)
    if pd.isna(base_ts):
        return False

    checks = []

    if not pd.isna(row.get("_ParentSig_dt", pd.NaT)):
        checks.append(
            within_time_tol_ignoring_date(row["_ParentSig_dt"], base_ts, SIGN_TOL_MIN)
        )

    if not pd.isna(row.get("_UserSig_dt", pd.NaT)):
        checks.append(
            within_time_tol_ignoring_date(row["_UserSig_dt"], base_ts, SIGN_TOL_MIN)
        )

    if not checks:
        return True
    return all(checks)

check_signatures = merged.apply(sig_times_ok, axis=1)

merged["_OverallPass"] = (
    check_duration
    & check_locations
    & check_billing
    & check_signatures
    & check_geo
)

# ---------- Clean up time display (AM/PM) ----------
time_display_cols = [
    "Start time",
    "End time",
    "Appt. Start Time",
    "Appt. End Time",
    "Parent signature time",
    "User signature time",
]

for col in time_display_cols:
    if col in merged.columns:
        merged[col] = pd.to_datetime(merged[col], errors="coerce").dt.strftime("%I:%M:%S %p")

# ---------- Computed display fields ----------
merged["Δ vs Billing (minutes)"] = merged["Actual Minutes"] - merged["Scheduled Minutes"]
merged["Δ vs Billing (HH:MM:SS)"] = merged["Δ vs Billing (minutes)"].apply(minutes_to_hhmmss_signed)

# ---------- Display ----------
display_cols = [
    "Status",
    "Appt. Date",
    "Appointment ID",
    "Client",
    "Appointment Location",
    "Staff Name",
    "Appt. Start Time",
    "Appt. End Time",
    "Start time",
    "End time",
    "Duration",
    "Location (start)",
    "Location (end)",
    "User signature location",
    "User signature time",
    "Parent signature location",
    "Parent signature time",
    "Service Name",
    "Scheduled Minutes",
    "Actual Minutes",
    "Δ vs Billing (HH:MM:SS)",
    "Geo User↔Appt ft",
    "Geo Parent↔Appt ft",
    "Geo Reason",
]
present_cols = [c for c in display_cols if c in merged.columns]

st.subheader("3) Results")
st.dataframe(merged[present_cols], use_container_width=True, height=560)

# ---------- Summary ----------
st.caption("Summary")
summary = pd.DataFrame({
    "Total (after filters)": [len(merged)],
    "Pass": [int(merged["_OverallPass"].sum())],
    "Fail": [int((~merged["_OverallPass"]).sum())],
})
st.table(summary)

# ---------- Export ----------
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
- Duration strictly >= {MIN_MINUTES} minutes
- **All selected location fields** must have content
- Billing within ±{BILLING_TOL_MIN} minutes (see signed Δ column)
- **Parent signature time** and **User signature time** must each be within ±{SIGN_TOL_MIN} minutes of the appointment end time ("Appt. End Time")
- **Parent/User signature locations** must each be within **{GEO_TOL_FT} ft** of **Appointment Location**
- Geo diagnostics available via **Geo Status** and **Geo Reason** columns
"""
)
