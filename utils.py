import re
import unicodedata
from typing import Any

import pandas as pd


# =========================================================
# Session Time helpers (same behavior as your imported utils)
# =========================================================
_TIME_RANGE_ANYWHERE_RE = re.compile(
    r"(?P<s_h>\d{1,2})\s*:\s*(?P<s_m>\d{2})"
    r"(?:\s*(?P<s_ampm>am|pm|AM|PM))?"
    r"\s*-\s*"
    r"(?P<e_h>\d{1,2})\s*:\s*(?P<e_m>\d{2})"
    r"(?:\s*(?P<e_ampm>am|pm|AM|PM))?"
)
def normalize_session_time(raw: str) -> str:
    """
    Canonical output: 'HH:MM AM - HH:MM PM'

    Rules:
    - 24-hour values (>=13) are converted normally
    - For 1–12 values without AM/PM:
        * If any hour >= 12 → PM
        * Otherwise → AM
    - Same-day only, no overnight inference
    """
    if not raw:
        return ""

    s = unicodedata.normalize("NFKC", str(raw))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Co")
    s = s.replace("\u00a0", " ").strip()
    if not s:
        return ""

    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s)

    m = _TIME_RANGE_ANYWHERE_RE.search(s.lower())
    if not m:
        return ""

    sh = int(m.group("s_h"))
    sm = int(m.group("s_m"))
    eh = int(m.group("e_h"))
    em = int(m.group("e_m"))
    s_ampm = (m.group("s_ampm") or "").lower()
    e_ampm = (m.group("e_ampm") or "").lower()

    if sm > 59 or em > 59:
        return ""

    def fmt(h, m, ap):
        return f"{h:02d}:{m:02d} {ap}"

    def from_24h(h):
        ap = "AM" if h < 12 else "PM"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        return h12, ap

    # --- 24-hour input ---
    if sh >= 13 or eh >= 13:
        if sh > 23 or eh > 23:
            return ""
        sh12, sap = from_24h(sh)
        eh12, eap = from_24h(eh)
        return f"{fmt(sh12, sm, sap)} - {fmt(eh12, em, eap)}"

    # --- 12-hour input ---
    # Explicit AM/PM wins
    if s_ampm and not e_ampm:
        e_ampm = s_ampm
    if e_ampm and not s_ampm:
        s_ampm = e_ampm

    # No AM/PM → hard business rule: PM for both
    if not s_ampm and not e_ampm:
        s_ampm = e_ampm = "pm"


    if not (1 <= sh <= 12 and 1 <= eh <= 12):
        return ""

    return f"{fmt(sh, sm, s_ampm.upper())} - {fmt(eh, em, e_ampm.upper())}"



def parse_session_time_range(session_time: Any, base_date: Any):
    """
    Business rules:
    - Same-day sessions only
    - Missing AM/PM already normalized to AM → PM
    - Latest allowed end time is 10:00 PM
    """
    if pd.isna(session_time):
        return pd.NaT, pd.NaT

    text = str(session_time).strip()
    if "-" not in text:
        return pd.NaT, pd.NaT

    start_str, end_str = [p.strip() for p in text.split("-", 1)]

    if not base_date:
        return pd.NaT, pd.NaT

    start_dt = pd.to_datetime(f"{base_date} {start_str}", errors="coerce")
    end_dt = pd.to_datetime(f"{base_date} {end_str}", errors="coerce")

    if pd.isna(start_dt) or pd.isna(end_dt):
        return pd.NaT, pd.NaT

    # ❌ No overnight sessions allowed
    if end_dt <= start_dt:
        return pd.NaT, pd.NaT

# ❌ Hard cutoff: before 10:08 PM
    if end_dt.hour > 22 or (end_dt.hour == 22 and end_dt.minute >= 8):
        return pd.NaT, pd.NaT


    return start_dt, end_dt



def normalize_date(raw: str) -> str:
    """
    Accepts:
      - 2025/10/28  (YYYY/MM/DD)
      - 10/28/2025  (MM/DD/YYYY)
      - 2025-10-28  (YYYY-MM-DD)  [optional support]
    Returns normalized 'MM/DD/YYYY' when possible, else the stripped raw string.
    """
    if not raw:
        return ""

    raw = str(raw).strip()
    if not raw:
        return ""
    # --- European format: DD.MM.YYYY ---
    m_dot = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", raw)
    if m_dot:
        d, m, y = map(int, m_dot.groups())
        return f"{m:02d}/{d:02d}/{y:04d}"
    # Optional: support YYYY-MM-DD too
    m_dash = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", raw)
    if m_dash:
        y, mo, d = map(int, m_dash.groups())
        return f"{mo:02d}/{d:02d}/{y:04d}"

    m = re.match(r"^(\d{1,4})/(\d{1,2})/(\d{1,4})$", raw)
    if not m:
        return raw  # unexpected format, return as-is

    a, b, c = m.groups()
    a_i, b_i, c_i = int(a), int(b), int(c)

    # Case 1: YYYY/MM/DD
    if len(a) == 4:
        year, month, day = a_i, b_i, c_i
    # Case 2: MM/DD/YYYY
    elif len(c) == 4:
        year, month, day = c_i, a_i, b_i
    else:
        # fallback: assume last is year
        year, month, day = c_i, a_i, b_i

    return f"{month:02d}/{day:02d}/{year:04d}"



# =========================================================
# PDF TEXT CLEANING + TIME NORMALIZATION (for parser)
# =========================================================
def strip_private_use(s: str) -> str:
    """Remove private-use unicode chars (where the '􀀀' junk usually lives)."""
    if s is None:
        return ""
    s = str(s)

    # ✅ NEW: remove the PDF "replacement" character (shows as ��)
    s = s.replace("\ufffd", "")

    return "".join(ch for ch in s if unicodedata.category(ch) != "Co")



def clean_time_text(s: str) -> str:
    """Normalize PDF-extracted time text into something parseable."""
    if not s:
        return ""
    s = strip_private_use(s)

    # normalize unicode width and punctuation
    s = unicodedata.normalize("NFKC", s)

    # unify dash variants
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")

    # normalize AM/PM variants
    s_low = s.lower()
    s_low = s_low.replace("a.m.", "am").replace("p.m.", "pm")
    s_low = s_low.replace("a. m.", "am").replace("p. m.", "pm")
    s = s_low

    # clean spacing around hyphen
    s = re.sub(r"\s*-\s*", " - ", s)

    # collapse whitespace
    s = " ".join(s.split()).strip()
    return s


_TIME_RANGE_RE = re.compile(
    r"(?P<s_h>\d{1,2})\s*:\s*(?P<s_m>\d{2})\s*(?P<s_ampm>am|pm)?"
    r"\s*-\s*"
    r"(?P<e_h>\d{1,2})\s*:\s*(?P<e_m>\d{2})\s*(?P<e_ampm>am|pm)?",
    re.I
)


def extract_first_time_range(raw: str) -> str:
    """
    Returns a normalized time range string like:
      '02:30 PM - 06:30 PM' or '10:00 AM - 02:00 PM'

    Mirrored rules:
    - If any hour >= 13 → 24-hour time conversion
    - Else (1–12 only):
        * If AM/PM missing → PM for both
    - Same-day only, no overnight inference
    """
    raw = clean_time_text(raw)
    if not raw:
        return ""

    m = _TIME_RANGE_RE.search(raw)
    if not m:
        return ""

    sh = int(m.group("s_h"))
    sm = int(m.group("s_m"))
    eh = int(m.group("e_h"))
    em = int(m.group("e_m"))
    s_ampm = (m.group("s_ampm") or "").lower()
    e_ampm = (m.group("e_ampm") or "").lower()

    if sm > 59 or em > 59:
        return ""
    if sh > 23 or eh > 23:
        return ""

    def fmt(h, minute, ampm):
        return f"{h:02d}:{minute:02d} {ampm}"

    def from_24h(hour):
        ampm = "AM" if hour < 12 else "PM"
        h12 = hour % 12
        if h12 == 0:
            h12 = 12
        return h12, ampm

    # ✅ RULE 1: 24-hour input
    if sh >= 13 or eh >= 13:
        sh12, sA = from_24h(sh)
        eh12, eA = from_24h(eh)
        return f"{fmt(sh12, sm, sA)} - {fmt(eh12, em, eA)}"

    # ✅ RULE 2: 12-hour input
    # Propagate AM/PM if only one side has it
    if s_ampm and not e_ampm:
        e_ampm = s_ampm
    if e_ampm and not s_ampm:
        s_ampm = e_ampm

    # No AM/PM → hard business rule: PM for both
    if not s_ampm and not e_ampm:
        s_ampm = e_ampm = "pm"

    if not (1 <= sh <= 12 and 1 <= eh <= 12):
        return ""

    return f"{fmt(sh, sm, s_ampm.upper())} - {fmt(eh, em, e_ampm.upper())}"




