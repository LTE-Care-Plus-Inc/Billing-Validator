import pandas as pd
from datetime import datetime


from utils import (
    normalize_session_time,
    parse_session_time_range,
    normalize_date,
    extract_first_time_range
)

# =====================================================
# normalize_session_time
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
