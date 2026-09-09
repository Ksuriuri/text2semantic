"""Regression cases accumulated in #TTS研究:7ef45303.

Add every future production normalizer report here before changing a rule. The
expected strings are the exact text passed into the T2S model, so this suite
tests classification and precedence without needing a GPU or audio model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import t2s_text_normalizer as normalizer  # noqa: E402


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "四十四：十四似四十，\t四十似十四，事实是十四，不是四十四。",
            "四十四：十四似四十，四十似十四，事实是十四，不是四十四。",
        ),
        ("你好\t世界", "你好，世界"),
        ("你好    世界", "你好，世界"),
        ("hello\tworld", "hello world"),
        ("你好，\t世界", "你好，世界"),
    ],
)
def test_tab_and_indent_rules(raw: str, expected: str) -> None:
    assert normalizer.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("KB, MB, GB, TB", "kilobyte, megabyte, gigabyte, terabyte"),
        ("16GB", "sixteen gigabytes"),
        ("1 GB", "one gigabyte"),
        ("1.5TB", "one point five terabytes"),
        ("100MB/s", "one hundred megabytes per second"),
        ("1 Gbps", "one gigabit per second"),
        ("2.4GHz", "two point four gigahertz"),
        ("89M users", "eighty-nine million users"),
        ("5G / 4K", "5G / 4K"),
        ("$89M", "eighty-nine million dollars"),
        ("$30", "thirty dollars"),
        ("$30.01", "thirty dollars and one cent"),
        ("$30.16", "thirty dollars and sixteen cents"),
        ("$1.16", "one dollar and sixteen cents"),
        ("€30.01", "thirty euros and one cent"),
        ("€1.16", "one euro and sixteen cents"),
        ("£30.01", "thirty pounds and one penny"),
        ("£1.16", "one pound and sixteen pence"),
    ],
)
def test_units_and_money_rules(raw: str, expected: str) -> None:
    assert normalizer.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "8-12只小狗，3%-5%的小猫",
            "八到十二只小狗，百分之三到百分之五的小猫",
        ),
        ("3-5天", "三到五天"),
        ("重量10-20kg", "重量十到二十 kg"),
        ("第3-5章", "第三到五章"),
        ("2024-2026年", "2024年到2026年"),
        ("9:00-10:30", "九点到十点三十分"),
        ("范围A-Z", "范围A到Z"),
        ("3%-5%", "百分之三到百分之五"),
        ("3-5%", "百分之三到百分之五"),
        ("3%~5%", "百分之三到百分之五"),
        ("3％—5％", "百分之三到百分之五"),
        ("3%至5%", "百分之三到百分之五"),
        ("3.5%", "百分之三点五"),
        ("100%", "百分之百"),
        ("下降1%", "下降百分之一"),
    ],
)
def test_ranges_and_percentages_in_chinese(raw: str, expected: str) -> None:
    assert normalizer.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2-1=1", "二减一等于一"),
        ("2 - 1", "二减一"),
        ("5%-3%=2%", "百分之五减百分之三等于百分之二"),
        ("-5", "负五"),
        ("-3%", "负百分之三"),
        ("温度是-5°C", "温度是零下五摄氏度"),
        ("温度是-5°F", "温度是零下五华氏度"),
    ],
)
def test_math_negatives_and_temperature(raw: str, expected: str) -> None:
    assert normalizer.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The range is 3%-5%.", "The range is three to five percent."),
        ("Calculate 2-1=1.", "Calculate two minus one equals one."),
        ("temperature -5°C", "temperature minus five degrees Celsius"),
        ("Use chapters 3-5 chapters", "Use chapters three to five chapters"),
        ("Use A-Z", "Use A to Z"),
        ("state-of-the-art real-time", "state of the art real time"),
    ],
)
def test_english_hyphen_and_percent_rules(raw: str, expected: str) -> None:
    assert normalizer.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("日期是2026-09-02", "日期是2026年9月2日"),
        ("编号G-123456", "编号G 一二三四五六"),
        ("G-123456", "G dash one two three four five six"),
        ("RTX-5090", "RTX dash five zero nine zero"),
        ("H100-SXM", "H one zero zero dash SXM"),
        ("010-1234-5678", "零一零 一二三四 五六七八"),
        ("你好—世界", "你好，世界"),
    ],
)
def test_dates_identifiers_and_dashes(raw: str, expected: str) -> None:
    assert normalizer.normalize(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "https://example.com/a-b?q=x%20y",
        "user-name@example.com",
        "/home/a-b/file",
        r"C:\data\a-b\file.wav",
        "--port",
        "-x",
        "v1.2.3",
        "%s",
        "%04d",
        "10 % 3",
    ],
)
def test_structured_syntax_is_protected(raw: str) -> None:
    assert normalizer.normalize(raw) == raw


@pytest.mark.parametrize("raw", ["3-1", "比分3-1结束", "3-1/2"])
def test_ambiguous_numeric_hyphens_are_deferred(raw: str) -> None:
    assert normalizer.normalize(raw) == raw


def test_describe_reports_each_rule_family() -> None:
    raw = "你好\t世界，8-12只，$30.16"
    normalized = normalizer.normalize(raw)
    assert normalizer.describe(raw, normalized) == (
        "tab/indent stripped; hyphen/percent expanded; units/money expanded"
    )


def test_none_is_empty() -> None:
    assert normalizer.normalize(None) == ""
