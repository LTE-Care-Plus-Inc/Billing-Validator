import re
import unicodedata
from typing import Any
import pandas as pd

_TIME_RANGE_ANYWHERE_RE = re.compile(
    r"(?P<s_h>\d{1,2})\s*:\s*(?P<s_m>\d{2})\s*(?P<s_ampm>am|pm|AM|PM)?"
    r"\s*-\s*"
    r"(?P<e_h>\d{1,2})\s*:\s*(?P<e_m>\d{2})\s*(?P<e_ampm>am|pm|AM|PM)?"
)

def normalize_session_time(raw: str) -> str:
    """
    Normalize session time into canonical form:
      'HH:MM AM - HH:MM PM'  (HH is zero-padded)
    Returns empty string if we cannot confidently parse a time range.
    """
    if raw is None:
        return ""

    s = unicodedata.normalize("NFKC", str(raw))
    # remove private-use unicode (where 􀀀 junk often lives)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Co")

    s = s.replace("\u00a0", " ").strip()
    if not s:
        return ""

    # Normalize dashes & spacing
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s)

    # Normalize a.m./p.m. variants -> am/pm
    s_low = s.lower()
    s_low = re.sub(r"\b(a\s*\.?\s*m\s*\.?)\b", "am", s_low)
    s_low = re.sub(r"\b(p\s*\.?\s*m\s*\.?)\b", "pm", s_low)
    s = s_low

    # Pull FIRST time range from anywhere in the string
    m = _TIME_RANGE_ANYWHERE_RE.search(s)
    if not m:
        return ""

    sh = int(m.group("s_h"))
    sm = int(m.group("s_m"))
    eh = int(m.group("e_h"))
    em = int(m.group("e_m"))
    s_ampm = (m.group("s_ampm") or "").lower()
    e_ampm = (m.group("e_ampm") or "").lower()

    # validate minutes
    if sm > 59 or em > 59:
        return ""

    # If only one side has am/pm, apply it to both
    if s_ampm and not e_ampm:
        e_ampm = s_ampm
    if e_ampm and not s_ampm:
        s_ampm = e_ampm

    def hour_to_12h(hour_0_23: int):
        if hour_0_23 < 0 or hour_0_23 > 23:
            return None
        ampm = "AM" if hour_0_23 < 12 else "PM"
        h12 = hour_0_23 % 12
        if h12 == 0:
            h12 = 12
        return h12, ampm

    # If neither has am/pm, infer (HiRasmus PDF common case)
    if not s_ampm and not e_ampm:
        # Conservative heuristic: after-school/evening tends to be PM
        if sh == 12 or sh >= 4:
            s_ampm = e_ampm = "pm"
        else:
            s_ampm = e_ampm = "am"

    # If hour looks like 24h (e.g., 13:00) and am/pm is missing (rare), convert via 24h logic
    # Otherwise treat as 12h input.
    def to_display(h: int, minute: int, ampm: str):
        # if input hour is 0-23 and we have am/pm, treat as 12h-style hour (1-12),
        # but still allow 0/13+ by converting safely.
        if h > 12 or h == 0:
            conv = hour_to_12h(h)
            if not conv:
                return None
            h12, inferred_ampm = conv
            # if ampm given, keep it; else use inferred
            final_ampm = ampm.upper() if ampm else inferred_ampm
            return f"{h12:02d}:{minute:02d} {final_ampm}"
        else:
            return f"{h:02d}:{minute:02d} {ampm.upper()}"

    left = to_display(sh, sm, s_ampm)
    right = to_display(eh, em, e_ampm)
    if not left or not right:
        return ""

    return f"{left} - {right}"


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
