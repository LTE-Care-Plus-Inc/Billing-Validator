import re
import unicodedata
from typing import Any, Tuple

import pandas as pd

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

def strip_private_use(s: str) -> str:
    """Remove private-use unicode chars and replacement char."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufffd", "")
    return "".join(ch for ch in s if unicodedata.category(ch) != "Co")


def clean_time_text(s: str) -> str:
    """
    Normalize extracted time text into something parse-friendly:
    - NFKC normalize
    - Chinese 上午/下午 -> am/pm
    - unify dash variants
    - standardize spacing around '-'
    - normalize a.m./p.m. variants
    """
    if not s:
        return ""

    s = strip_private_use(s)

    # normalize unicode width and punctuation
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")

    # Chinese markers -> am/pm
    s = s.replace("上午", "am").replace("下午", "pm")

    # unify dash variants
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")

    # normalize AM/PM variants and case
    s_low = s.lower()
    s_low = s_low.replace("a.m.", "am").replace("p.m.", "pm")
    s_low = s_low.replace("a. m.", "am").replace("p. m.", "pm")
    s = s_low

    # standardize spacing around hyphen
    s = re.sub(r"\s*-\s*", " - ", s)

    # collapse whitespace
    s = " ".join(s.split()).strip()
    return s


def normalize_time_range(raw):
    # If already canonical, never reinterpret
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

    # ✅ ONLY these groups exist now
    s_ampm = (m.group("s_ampm1") or m.group("s_ampm2") or "").lower()
    e_ampm = (m.group("e_ampm1") or m.group("e_ampm2") or "").lower()

    def fmt(h12, m, ap):
        return f"{h12:02d}:{m:02d} {ap}"

    def from_24h(h):
        ap = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return h12, ap

    # ---- Rule 1: explicit 24-hour ----
    if sh >= 13 or eh >= 13:
        sh12, sap = from_24h(sh)
        eh12, eap = from_24h(eh)
        return f"{fmt(sh12, sm, sap)} - {fmt(eh12, em, eap)}"

    # ---- Rule 2: explicit AM + PM on both sides ----
    if s_ampm and e_ampm:
        return f"{fmt(sh, sm, s_ampm.upper())} - {fmt(eh, em, e_ampm.upper())}"

    # ---- Rule 3: propagate single-sided AM/PM ----
    if s_ampm and not e_ampm:
        e_ampm = s_ampm
    if e_ampm and not s_ampm:
        s_ampm = e_ampm

    if s_ampm and e_ampm:
        return f"{fmt(sh, sm, s_ampm.upper())} - {fmt(eh, em, e_ampm.upper())}"
    # ---- Rule 3.5: corrupted Chinese AM/PM inference (�� / �) ----
    raw_s = "" if raw is None else str(raw)
    if ("\ufffd" in raw_s) or ("��" in raw_s) or ("�" in raw_s):
        # If end hour < start hour, treat as morning -> afternoon
        # e.g. 9:00 - 3:00  => 9:00 AM - 3:00 PM
        if eh < sh:
            return f"{fmt(sh, sm, 'AM')} - {fmt(eh, em, 'PM')}"

        # Otherwise assume afternoon block for typical ABA ranges
        # e.g. 2:30 - 6:00  => 2:30 PM - 6:00 PM
        # Keep 11-12 edge safe: 11-12 could be AM, but most BT sessions are PM; adjust if needed.
        if 1 <= sh <= 7 and 3 <= eh <= 10:
            return f"{fmt(sh, sm, 'PM')} - {fmt(eh, em, 'PM')}"

    # ---- Rule 4: ONLY if no AM/PM tokens existed in original string ----
    if "am" not in s and "pm" not in s:
        sh12, sap = from_24h(sh)
        eh12, eap = from_24h(eh)
        return f"{fmt(sh12, sm, sap)} - {fmt(eh12, em, eap)}"

    return ""




def normalize_session_time(raw: str) -> str:
    """
    Backwards-compatible wrapper.
    Previously your app called normalize_session_time(raw) during merge.
    Now it uses the SINGLE authoritative normalization.
    """
    return normalize_time_range(raw)


def extract_first_time_range(raw: str) -> str:
    """
    Backwards-compatible wrapper.
    Previously PDF parser called extract_first_time_range(raw_session_time).
    Now it uses the SINGLE authoritative normalization.
    """
    return normalize_time_range(raw)


def parse_session_time_range(session_time: Any, base_date: Any):
    """
    Business rules:
    - Same-day sessions only
    - No overnight sessions
    - Hard cutoff: end must be before 10:08 PM
    NOTE: This function assumes session_time is already canonical or empty.
    """
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

    # No overnight sessions allowed
    if end_dt <= start_dt:
        return pd.NaT, pd.NaT

    # Hard cutoff: before 10:08 PM
    if end_dt.hour > 22 or (end_dt.hour == 22 and end_dt.minute >= 8):
        return pd.NaT, pd.NaT

    return start_dt, end_dt


def normalize_date(raw: str) -> str:
    """
    Normalize dates to MM/DD/YYYY.

    Supported:
    - DD.MM.YYYY
    - DD/MM/YYYY (if day > 12)
    - MM/DD/YYYY
    - YYYY-MM-DD, YYYY/MM/DD
    """
    if raw is None:
        return ""

    raw = str(raw).strip()
    if not raw:
        return ""

    # ISO formats: YYYY-MM-DD or YYYY/MM/DD
    m_iso = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", raw)
    if m_iso:
        y, m, d = map(int, m_iso.groups())
        return f"{m:02d}/{d:02d}/{y:04d}"

    # European dotted: DD.MM.YYYY
    m_dot = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", raw)
    if m_dot:
        d, m, y = map(int, m_dot.groups())
        return f"{m:02d}/{d:02d}/{y:04d}"

    # Slash: MM/DD/YYYY or DD/MM/YYYY
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
