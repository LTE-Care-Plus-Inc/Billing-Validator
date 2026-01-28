import pandas as pd

from ..utils import (
    normalize_session_time,
    extract_first_time_range,
    parse_session_time_range,
)

# =====================================================
# normalize_session_time — existing coverage
# =====================================================

def test_normalize_24_hour_basic():
    assert normalize_session_time("14:30 - 18:30") == "02:30 PM - 06:30 PM"
    assert normalize_session_time("10:00 - 14:00") == "10:00 AM - 02:00 PM"
    assert normalize_session_time("09:15 - 13:45") == "09:15 AM - 01:45 PM"

def test_normalize_explicit_ampm():
    assert normalize_session_time("9:00 am - 1:00 pm") == "09:00 AM - 01:00 PM"
    assert normalize_session_time("2:00 pm - 6:00 pm") == "02:00 PM - 06:00 PM"


def test_normalize_whitespace_and_unicode_dashes():
    assert normalize_session_time(" 9:00 am – 1:00 pm ") == "09:00 AM - 01:00 PM"
    assert normalize_session_time("2:00pm—6:00pm") == "02:00 PM - 06:00 PM"


def test_normalize_invalid():
    assert normalize_session_time("") == ""
    assert normalize_session_time("hello") == ""
    assert normalize_session_time("25:00 - 26:00") == ""


# =====================================================
# ✅ Chinese / bilingual time strings
# =====================================================

def test_normalize_chinese_ampm_words():
    # 上午/下午
    assert normalize_session_time("上午9:00 - 下午1:00") == "09:00 AM - 01:00 PM"
    assert normalize_session_time("下午2:00 - 下午6:00") == "02:00 PM - 06:00 PM"


def test_normalize_chinese_noon_word():
    # 中午 commonly means PM; 12:00 中午 → 12:00 PM
    assert normalize_session_time("中午12:00 - 下午3:00") == "12:00 PM - 03:00 PM"


def test_normalize_chinese_fullwidth_colon_and_dash():
    # Fullwidth colon ： and long dash —
    assert normalize_session_time("14：30 — 18：30") == "02:30 PM - 06:30 PM"


def test_normalize_chinese_with_prefix_text():
    assert normalize_session_time("时间：下午2:00 - 下午6:00（家长在场）") == "02:00 PM - 06:00 PM"


# =====================================================
# parse_session_time_range — existing coverage
# =====================================================

def test_parse_valid_same_day():
    start, end = parse_session_time_range("02:30 PM - 06:30 PM", "01/15/2026")
    assert start == pd.Timestamp("2026-01-15 14:30")
    assert end == pd.Timestamp("2026-01-15 18:30")


def test_parse_zero_length_session():
    start, end = parse_session_time_range("02:00 PM - 02:00 PM", "01/15/2026")
    assert pd.isna(start)
    assert pd.isna(end)


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


def test_parse_just_before_10pm():
    start, end = parse_session_time_range("07:00 PM - 09:59 PM", "01/15/2026")
    assert start == pd.Timestamp("2026-01-15 19:00")
    assert end == pd.Timestamp("2026-01-15 21:59")


# =====================================================
# extract_first_time_range — existing + Chinese
# =====================================================

def test_extract_24_hour():
    assert extract_first_time_range("14:30 - 18:30") == "02:30 PM - 06:30 PM"
    assert extract_first_time_range("10:00 - 14:00") == "10:00 AM - 02:00 PM"


def test_extract_explicit_ampm():
    assert extract_first_time_range("9:00 am - 1:00 pm") == "09:00 AM - 01:00 PM"


def test_extract_first_of_multiple_ranges():
    text = "Session ran 9:00 am - 11:00 am, then 1:00 pm - 3:00 pm"
    assert extract_first_time_range(text) == "09:00 AM - 11:00 AM"


def test_extract_chinese_first_range():
    text = "时间：下午2:00 - 下午6:00（备注：孩子表现良好）"
    assert extract_first_time_range(text) == "02:00 PM - 06:00 PM"


def test_extract_invalid():
    assert extract_first_time_range("") == ""
    assert extract_first_time_range("no time here") == ""
    assert extract_first_time_range("这里没有时间") == ""
