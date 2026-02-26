# ===========================
# utils.py
# ===========================
import io
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Excel-safe sanitizer
# =========================================================
_ILLEGAL_XL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def excel_safe_str(v: Any) -> Any:
    if v is None:
        return v
    if isinstance(v, float) and pd.isna(v):
        return v
    s = str(v)
    s = _ILLEGAL_XL_CHARS_RE.sub("", s)
    return s


def excel_sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == object:
            df2[c] = df2[c].map(excel_safe_str)
    return df2


def export_excel(df: pd.DataFrame) -> bytes:
    df2 = excel_sanitize_df(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df2.to_excel(w, index=False, sheet_name="All")
    return buf.getvalue()


# =========================================================
# File reading
# =========================================================
def read_any(file):
    if file is None:
        return None
    if file.name.lower().endswith((".csv", ".txt")):
        return pd.read_csv(file, dtype=str)
    return pd.read_excel(file, dtype=str, engine="openpyxl")


# =========================================================
# DataFrame helpers
# =========================================================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
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


# =========================================================
# Duration parsing
# =========================================================
def parse_duration_to_minutes(d: Any) -> float:
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
    return np.nan


# =========================================================
# Name normalization
# =========================================================
def normalize_name(name: Any) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip()
    if not s:
        return ""
    parts = s.replace(",", " ").split()
    parts = [p.capitalize() for p in parts if p]
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    last = parts[-1]
    return f"{last}, {first}"


def normalize_client_name_for_match(name: Any) -> str:
    return normalize_name(name)


# =========================================================
# Time helpers
# =========================================================
def within_time_tol(sig_ts, base_ts, tol_early_min):
    if pd.isna(sig_ts) or pd.isna(base_ts):
        return False
    diff_min = (sig_ts - base_ts).total_seconds() / 60.0
    return diff_min > tol_early_min


# =========================================================
# Time normalization (kept here for backwards compatibility)
# =========================================================
def strip_private_use(s: str) -> str:
    """Remove private-use unicode chars and replacement char."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufffd", "")
    return "".join(ch for ch in s if unicodedata.category(ch) != "Co")


def clean_time_text(s: str) -> str:
    if not s:
        return ""

    s = str(s)

    # Step 1: infer AM/PM from corrupted markers BEFORE stripping
    def _infer_ampm(m):
        rest = s[m.end():]
        hour_match = re.match(r"\s*(\d{1,2})\s*:", rest)
        if not hour_match:
            return ""
        h = int(hour_match.group(1))
        if h == 12 or 1 <= h <= 7:
            return "pm "
        return "am "

    s = re.sub(r"\ufffd+", _infer_ampm, s)

    # Step 2: now safe to strip remaining junk
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")

    # Chinese markers (if intact)
    s = s.replace("上午", "am ").replace("下午", "pm ")

    # unify dash variants
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")

    # normalize AM/PM variants
    s = s.lower()
    s = s.replace("a.m.", "am").replace("p.m.", "pm")
    s = s.replace("a. m.", "am").replace("p. m.", "pm")

    # standardize spacing around hyphen
    s = re.sub(r"\s*-\s*", " - ", s)

    # collapse whitespace
    s = " ".join(s.split()).strip()
    return s


_TIME_RANGE_RE = re.compile(
    r"(?:(?P<s_ampm1>am|pm)\s*)?"
    r"(?P<s_h>\d{1,2})\s*:\s*(?P<s_m>\d{2})"
    r"\s*(?P<s_ampm2>am|pm)?"
    r"\s*-\s*"
    r"(?:(?P<e_ampm1>am|pm)\s*)?"
    r"(?P<e_h>\d{1,2})\s*:\s*(?P<e_m>\d{2})"
    r"\s*(?P<e_ampm2>am|pm)?",
    re.I,
)


def normalize_time_range(raw):
    if isinstance(raw, str) and re.search(
        r"\b\d{1,2}:\d{2}\s*(AM|PM)\s*-\s*\d{1,2}:\d{2}\s*(AM|PM)\b",
        raw
    ):
        return raw.strip()

    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""

    s = clean_time_text(str(raw))
    if not s:
        return ""

    m = _TIME_RANGE_RE.search(s)
    if not m:
        return ""

    sh = int(m.group("s_h"))
    sm = int(m.group("s_m"))
    eh = int(m.group("e_h"))
    em = int(m.group("e_m"))

    if sm > 59 or em > 59 or sh > 23 or eh > 23:
        return ""

    s_ampm = (m.group("s_ampm1") or m.group("s_ampm2") or "").lower()
    e_ampm = (m.group("e_ampm1") or m.group("e_ampm2") or "").lower()

    def fmt(h12, m, ap):
        return f"{h12:02d}:{m:02d} {ap}"

    def from_24h(h):
        ap = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return h12, ap

    if sh >= 13 or eh >= 13:
        sh12, sap = from_24h(sh)
        eh12, eap = from_24h(eh)
        return f"{fmt(sh12, sm, sap)} - {fmt(eh12, em, eap)}"

    if s_ampm and e_ampm:
        sh_out = sh if sh != 0 else 12
        eh_out = eh if eh != 0 else 12
        return f"{fmt(sh_out, sm, s_ampm.upper())} - {fmt(eh_out, em, e_ampm.upper())}"

    if s_ampm and not e_ampm:
        e_ampm = s_ampm
    if e_ampm and not s_ampm:
        s_ampm = e_ampm

    if s_ampm and e_ampm:
        return f"{fmt(sh, sm, s_ampm.upper())} - {fmt(eh, em, e_ampm.upper())}"

    raw_s = "" if raw is None else str(raw)
    if ("\ufffd" in raw_s) or ("��" in raw_s) or ("" in raw_s):
        if eh < sh:
            return f"{fmt(sh, sm, 'AM')} - {fmt(eh, em, 'PM')}"
        if 1 <= sh <= 7 and 3 <= eh <= 10:
            return f"{fmt(sh, sm, 'PM')} - {fmt(eh, em, 'PM')}"

    if "am" not in s and "pm" not in s:
        sh12, sap = from_24h(sh)
        eh12, eap = from_24h(eh)
        return f"{fmt(sh12, sm, sap)} - {fmt(eh12, em, eap)}"

    return ""


def normalize_session_time(raw: str) -> str:
    return normalize_time_range(raw)


def extract_first_time_range(raw: str) -> str:
    return normalize_time_range(raw)


def parse_session_time_range(session_time: Any, base_date: Any):
    if session_time is None or (isinstance(session_time, float) and pd.isna(session_time)):
        return pd.NaT, pd.NaT

    text = str(session_time).strip()
    if not text or "-" not in text:
        return pd.NaT, pd.NaT

    if not base_date:
        return pd.NaT, pd.NaT

    start_str, end_str = [p.strip() for p in text.split("-", 1)]

    start_dt = pd.to_datetime(f"{base_date} {start_str}", errors="coerce")
    end_dt = pd.to_datetime(f"{base_date} {end_str}", errors="coerce")

    if pd.isna(start_dt) or pd.isna(end_dt):
        return pd.NaT, pd.NaT

    if end_dt <= start_dt:
        return pd.NaT, pd.NaT

    if end_dt.hour > 22 or (end_dt.hour == 22 and end_dt.minute >= 8):
        return pd.NaT, pd.NaT

    return start_dt, end_dt


def normalize_date(raw: str) -> str:
    if raw is None:
        return ""

    raw = str(raw).strip()
    if not raw:
        return ""

    m_iso = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", raw)
    if m_iso:
        y, m, d = map(int, m_iso.groups())
        return f"{m:02d}/{d:02d}/{y:04d}"

    m_dot = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", raw)
    if m_dot:
        d, m, y = map(int, m_dot.groups())
        return f"{m:02d}/{d:02d}/{y:04d}"

    m_slash = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", raw)
    if m_slash:
        p1, p2, y = map(int, m_slash.groups())

        if p1 > 12 and 1 <= p2 <= 12:
            day, month = p1, p2
        elif p2 > 12 and 1 <= p1 <= 12:
            month, day = p1, p2
        else:
            month, day = p1, p2

        return f"{month:02d}/{day:02d}/{y:04d}"

    return raw