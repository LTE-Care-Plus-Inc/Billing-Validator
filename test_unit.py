import re
import pandas as pd

from utils import (
    normalize_session_time,
    normalize_time_range,
    parse_session_time_range,
    normalize_date,
    extract_first_time_range,
)


# =====================================================
# normalize_session_time — 24-hour input
# =====================================================

def test_normalize_24_hour_basic():
    assert normalize_session_time("14:30 - 18:30") == "02:30 PM - 06:30 PM"
    assert normalize_session_time("10:00 - 14:00") == "10:00 AM - 02:00 PM"
    assert normalize_session_time("09:15 - 13:45") == "09:15 AM - 01:45 PM"


def test_normalize_12_hour_missing_ampm():
    assert normalize_session_time("02:30 - 06:30") == "02:30 PM - 06:30 PM"
    assert normalize_session_time("12:00 - 3:00") == "12:00 PM - 03:00 PM"


def test_normalize_explicit_ampm():
    assert normalize_session_time("9:00 am - 1:00 pm") == "09:00 AM - 01:00 PM"
    assert normalize_session_time("2:00 pm - 6:00 pm") == "02:00 PM - 06:00 PM"


def test_normalize_invalid():
    assert normalize_session_time("") == ""
    assert normalize_session_time("hello") == ""
    assert normalize_session_time("25:00 - 26:00") == ""


# =====================================================
# Chinese time markers (下午 = PM, 上午 = AM)
# =====================================================

def test_chinese_pm_intact():
    """下午 = PM — when characters survive PDF extraction correctly."""
    result = normalize_time_range("下午6:00 - 下午9:00")
    assert result == "06:00 PM - 09:00 PM", f"Got: {result!r}"


def test_chinese_am_intact():
    """上午 = AM — when characters survive PDF extraction correctly."""
    result = normalize_time_range("上午9:00 - 上午11:00")
    assert result == "09:00 AM - 11:00 AM", f"Got: {result!r}"


def test_chinese_am_to_pm_cross_noon():
    """上午 start, 下午 end — crosses noon."""
    result = normalize_time_range("上午10:00 - 下午1:00")
    assert result == "10:00 AM - 01:00 PM", f"Got: {result!r}"


def test_chinese_pm_corrupted_flags_ambiguous():
    """
    When 下午 is corrupted to replacement chars (\\ufffd\\ufffd) the AM/PM
    signal is lost.  The normalizer should flag with ⚠️ and default to AM
    so Tab 2's signature resolver can correct it.
    """
    corrupted = "\ufffd\ufffd6:00 - \ufffd\ufffd9:00"
    result = normalize_time_range(corrupted)
    assert "⚠️" in result, f"Expected warning flag, got: {result!r}"
    assert "AM" in result, f"Expected AM default before signature resolution, got: {result!r}"


def test_chinese_pm_corrupted_single_replacement():
    """Single replacement char variant — still flags ambiguous."""
    corrupted = "\ufffd6:00 - \ufffd9:00"
    result = normalize_time_range(corrupted)
    assert "⚠️" in result, f"Expected warning flag, got: {result!r}"


# =====================================================
# parse_session_time_range
# =====================================================

def test_parse_valid_same_day():
    start, end = parse_session_time_range("02:30 PM - 06:30 PM", "01/15/2026")
    assert start == pd.Timestamp("2026-01-15 14:30")
    assert end == pd.Timestamp("2026-01-15 18:30")


def test_parse_reject_overnight():
    start, end = parse_session_time_range("10:00 PM - 02:00 PM", "01/15/2026")
    assert pd.isna(start)
    assert pd.isna(end)


def test_parse_reject_past_10pm():
    start, end = parse_session_time_range("07:00 PM - 10:30 PM", "01/15/2026")
    assert pd.isna(start)
    assert pd.isna(end)


def test_parse_accept_exact_10pm():
    start, end = parse_session_time_range("07:00 PM - 10:00 PM", "01/15/2026")
    assert start == pd.Timestamp("2026-01-15 19:00")
    assert end == pd.Timestamp("2026-01-15 22:00")


def test_parse_missing_inputs():
    assert parse_session_time_range(None, "01/15/2026") == (pd.NaT, pd.NaT)
    assert parse_session_time_range("02:30 PM - 06:30 PM", None) == (pd.NaT, pd.NaT)
    assert parse_session_time_range("", "01/15/2026") == (pd.NaT, pd.NaT)


# =====================================================
# normalize_date
# =====================================================

def test_normalize_date_us():
    assert normalize_date("10/28/2025") == "10/28/2025"
    assert normalize_date("2025/10/28") == "10/28/2025"


def test_normalize_date_european():
    assert normalize_date("13.01.2026") == "01/13/2026"
    assert normalize_date("01.12.2025") == "12/01/2025"


def test_normalize_date_iso():
    assert normalize_date("2025-10-28") == "10/28/2025"


def test_normalize_date_new_format():
    """Single-digit month/day as used in new note template (Date: 4/1/2026)."""
    assert normalize_date("4/1/2026") == "04/01/2026"
    assert normalize_date("3/31/2026") == "03/31/2026"


def test_normalize_date_invalid():
    assert normalize_date("") == ""
    assert normalize_date("abc") == "abc"


# =====================================================
# extract_first_time_range
# =====================================================

def test_extract_24_hour():
    assert extract_first_time_range("14:30 - 18:30") == "02:30 PM - 06:30 PM"
    assert extract_first_time_range("10:00 - 14:00") == "10:00 AM - 02:00 PM"


def test_extract_12_hour_missing_ampm():
    assert extract_first_time_range("2:30 - 6:30") == "02:30 PM - 06:30 PM"
    assert extract_first_time_range("12:00 - 3:00") == "12:00 PM - 03:00 PM"


def test_extract_explicit_ampm():
    assert extract_first_time_range("9:00 am - 1:00 pm") == "09:00 AM - 01:00 PM"


def test_extract_invalid():
    assert extract_first_time_range("") == ""
    assert extract_first_time_range("no time here") == ""


# =====================================================
# Session date regex — new vs old template labels
# =====================================================

DATE_RE = r"(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})"
_DATE_PATTERN = re.compile(rf"(?:Session\s+)?Date\b(?!\s+of\b)\s*:\s*{DATE_RE}", re.I)


def test_date_regex_old_label():
    """Old template: 'Session Date: 3/31/2026'"""
    m = _DATE_PATTERN.search("Session Date: 3/31/2026")
    assert m and m.group(1) == "3/31/2026"


def test_date_regex_new_label():
    """New template: bare 'Date: 4/1/2026'"""
    m = _DATE_PATTERN.search("Date: 4/1/2026")
    assert m and m.group(1) == "4/1/2026"


def test_date_regex_ignores_dob():
    """Must not match 'Date of Birth' as the session date."""
    block = "Date of Birth: 1/18/2020\nDate: 4/1/2026"
    m = _DATE_PATTERN.search(block)
    assert m and m.group(1) == "4/1/2026"


def test_date_regex_ignores_dob_only_block():
    """Block that only has Date of Birth — should return no session date."""
    block = "Date of Birth: 1/18/2020"
    m = _DATE_PATTERN.search(block)
    assert m is None


# =====================================================
# Session time regex — singular vs plural label
# =====================================================

_TIME_PATTERN = re.compile(r"(?im)^\s*Session Times?\s*:\s*(.*)\s*$")


def test_time_regex_old_label():
    """Old template: 'Session Time: 1:30 PM - 2:00 PM'"""
    m = _TIME_PATTERN.search("Session Time: 1:30 PM - 2:00 PM")
    assert m and m.group(1).strip() == "1:30 PM - 2:00 PM"


def test_time_regex_new_label():
    """New template: 'Session Times: 5:16 PM - 5:17 PM'"""
    m = _TIME_PATTERN.search("Session Times: 5:16 PM - 5:17 PM")
    assert m and m.group(1).strip() == "5:16 PM - 5:17 PM"


# =====================================================
# Attendees regex — old vs new label
# =====================================================

_BT_PATTERN = re.compile(
    r"\b(?:BT\s*/\s*RBT|RBT\s*/\s*BT|Behavior Technician\s*/\s*Registered Behavior Technician)\b",
    re.I,
)
_CAREGIVER_PATTERN = re.compile(r"\b(?:Adult Caregiver|Caregiver)\b", re.I)
_CLIENT_PATTERN = re.compile(r"\bClient\b", re.I)


def _get_present_text(block: str) -> str:
    lower = block.lower()
    pos = lower.find("present at session")
    if pos == -1:
        pos = lower.find("individuals present")
    return block[pos: pos + 400] if pos != -1 else ""


def test_attendees_old_label():
    block = "Present at Session: Behavior Technician / Registered Behavior Technician, Client, Caregiver"
    pt = _get_present_text(block)
    assert bool(_CLIENT_PATTERN.search(pt))
    assert bool(_BT_PATTERN.search(pt))
    assert bool(_CAREGIVER_PATTERN.search(pt))


def test_attendees_new_label():
    block = "Individuals Present: Behavior Technician / Registered Behavior Technician, Client, Caregiver"
    pt = _get_present_text(block)
    assert bool(_CLIENT_PATTERN.search(pt))
    assert bool(_BT_PATTERN.search(pt))
    assert bool(_CAREGIVER_PATTERN.search(pt))


def test_attendees_label_missing():
    """No attendee label — should find nothing."""
    block = "Maladaptive Status: N/A"
    pt = _get_present_text(block)
    assert pt == ""


# =====================================================
# Session Location — same-line vs next-line (table format)
# =====================================================

def _parse_location(block: str) -> str:
    loc_m = re.search(r"Session Location[ \t]*:[ \t]*([^\r\n]+)", block)
    session_location = loc_m.group(1).strip() if loc_m else ""
    if not session_location:
        loc_next = re.search(r"Session Location[ \t]*:\s*\n\s*([^\r\n]+)", block)
        if loc_next:
            candidate = loc_next.group(1).strip()
            if candidate and not candidate.endswith(":"):
                session_location = candidate
    return session_location


def test_location_same_line():
    block = "Session Location: 88 Northern Blvd, Flushing, New York 11021\nMaladaptive Status:"
    assert _parse_location(block) == "88 Northern Blvd, Flushing, New York 11021"


def test_location_next_line_table_format():
    """New table PDF format: label and value in separate cells → separate lines."""
    block = "Session Location:\n88 Northern Blvd, Flushing, New York 11021\nMaladaptive Status:"
    assert _parse_location(block) == "88 Northern Blvd, Flushing, New York 11021"


def test_location_empty_new_format():
    """Note 76 has no session location — should return empty (triggers compliance error)."""
    block = "Session Location:\nMaladaptive Status:"
    assert _parse_location(block) == ""


def test_location_does_not_capture_next_section_header():
    """Next line is a section header (ends with ':') — must not be captured as location."""
    block = "Session Location:\nMaladaptive Status:"
    assert _parse_location(block) == ""


# =====================================================
# Demo client filter
# =====================================================

_DEMO_PATTERN = re.compile(r"demo|marry", re.I)


def test_demo_filter_catches_normalized_name():
    """'Marry Wang Demo' normalizes to 'Demo, Marry' — both keywords present."""
    assert _DEMO_PATTERN.search("Demo, Marry")


def test_demo_filter_catches_raw_name():
    assert _DEMO_PATTERN.search("Marry Wang Demo")


def test_demo_filter_passes_real_client():
    assert not _DEMO_PATTERN.search("Wang, John")
    assert not _DEMO_PATTERN.search("Smith, Emily")


def test_demo_filter_case_insensitive():
    assert _DEMO_PATTERN.search("DEMO, MARRY")
    assert _DEMO_PATTERN.search("demo client")


# =====================================================
# Excel-based AM/PM resolution
# =====================================================

def _make_excel_dt(date_str, time_str):
    """Helper: build a pandas Timestamp from a date + time string."""
    return pd.Timestamp(f"{date_str} {time_str}")


def _resolve_ampm(time_str, excel_start, excel_end):
    """
    Mirror of the resolve_ampm logic in billing_checker.py so we can
    unit-test it independently.
    """
    import re as _re
    clean = time_str.replace("  ⚠️ AM/PM unknown", "").strip()

    if pd.notna(excel_start) and pd.notna(excel_end):
        m = _re.match(
            r"(\d{1,2}:\d{2})\s*AM\s*-\s*(\d{1,2}:\d{2})\s*AM",
            clean, _re.I,
        )
        if m:
            start_ap = "PM" if excel_start.hour >= 12 else "AM"
            end_ap   = "PM" if excel_end.hour   >= 12 else "AM"
            return f"{m.group(1)} {start_ap} - {m.group(2)} {end_ap}"

    return time_str  # unresolvable


def test_excel_resolver_pm_session():
    """Corrupted 6 PM – 9 PM session should be correctly resolved to PM."""
    ambiguous = "06:00 AM - 09:00 AM  ⚠️ AM/PM unknown"
    excel_start = _make_excel_dt("4/1/2026", "6:00 PM")
    excel_end   = _make_excel_dt("4/1/2026", "9:00 PM")
    result = _resolve_ampm(ambiguous, excel_start, excel_end)
    assert result == "06:00 PM - 09:00 PM", f"Got: {result!r}"


def test_excel_resolver_am_session():
    """Morning session (AM) should stay AM after resolution."""
    ambiguous = "09:00 AM - 11:00 AM  ⚠️ AM/PM unknown"
    excel_start = _make_excel_dt("4/1/2026", "9:00 AM")
    excel_end   = _make_excel_dt("4/1/2026", "11:00 AM")
    result = _resolve_ampm(ambiguous, excel_start, excel_end)
    assert result == "09:00 AM - 11:00 AM", f"Got: {result!r}"


def test_excel_resolver_no_excel_keeps_flag():
    """When Excel times are unavailable the ⚠️ flag must survive."""
    ambiguous = "06:00 AM - 09:00 AM  ⚠️ AM/PM unknown"
    result = _resolve_ampm(ambiguous, pd.NaT, pd.NaT)
    assert "⚠️" in result, f"Expected flag preserved, got: {result!r}"


# =====================================================
# PDF time vs Excel time accuracy check
# =====================================================

def _pdf_time_matches_excel(pdf_time, excel_start, excel_end):
    """Mirror of billing_checker._pdf_time_matches_excel for unit testing."""
    import re as _re
    import numpy as _np

    if not pdf_time or pdf_time == "nan" or "⚠️" in pdf_time:
        return _np.nan

    if pd.isna(excel_start) or pd.isna(excel_end):
        return _np.nan

    m = _re.match(
        r"(\d{1,2}):(\d{2})\s*(AM|PM)\s*-\s*(\d{1,2}):(\d{2})\s*(AM|PM)",
        pdf_time, _re.I,
    )
    if not m:
        return False

    def to_24h(h, ap):
        return int(h) % 12 + (12 if ap.upper() == "PM" else 0)

    return (
        to_24h(m.group(1), m.group(3)) == excel_start.hour
        and int(m.group(2))            == excel_start.minute
        and to_24h(m.group(4), m.group(6)) == excel_end.hour
        and int(m.group(5))            == excel_end.minute
    )


def test_pdf_time_matches_excel_exact():
    """PDF time matches Excel time exactly → True."""
    result = _pdf_time_matches_excel(
        "8:00 PM - 10:00 PM",
        _make_excel_dt("4/1/2026", "8:00 PM"),
        _make_excel_dt("4/1/2026", "10:00 PM"),
    )
    assert result is True


def test_pdf_time_wrong_hour():
    """BT documented the wrong hour → False."""
    result = _pdf_time_matches_excel(
        "7:00 PM - 10:00 PM",
        _make_excel_dt("4/1/2026", "8:00 PM"),
        _make_excel_dt("4/1/2026", "10:00 PM"),
    )
    assert result is False


def test_pdf_time_wrong_ampm():
    """BT documented AM instead of PM → False."""
    result = _pdf_time_matches_excel(
        "8:00 AM - 10:00 AM",
        _make_excel_dt("4/1/2026", "8:00 PM"),
        _make_excel_dt("4/1/2026", "10:00 PM"),
    )
    assert result is False


def test_pdf_time_still_ambiguous_returns_nan():
    """Unresolved ⚠️ flag → NaN (can't verify, don't penalise)."""
    import numpy as _np
    result = _pdf_time_matches_excel(
        "08:00 AM - 10:00 AM  ⚠️ AM/PM unknown",
        _make_excel_dt("4/1/2026", "8:00 PM"),
        _make_excel_dt("4/1/2026", "10:00 PM"),
    )
    assert _np.isnan(result)


def test_pdf_time_missing_returns_nan():
    """Empty PDF time → NaN (existence failure already caught elsewhere)."""
    import numpy as _np
    result = _pdf_time_matches_excel(
        "",
        _make_excel_dt("4/1/2026", "8:00 PM"),
        _make_excel_dt("4/1/2026", "10:00 PM"),
    )
    assert _np.isnan(result)


def test_pdf_time_no_excel_returns_nan():
    """No Excel time available → NaN (can't verify, don't penalise)."""
    import numpy as _np
    result = _pdf_time_matches_excel("8:00 PM - 10:00 PM", pd.NaT, pd.NaT)
    assert _np.isnan(result)
