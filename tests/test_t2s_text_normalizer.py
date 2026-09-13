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
        ("KB, MB, GB, TB", 'KB, MB, GB, TB'),
        ("16GB", '16GB'),
        ("1 GB", '1 GB'),
        ("1.5TB", '1.5TB'),
        ("100MB/s", '100MB/s'),
        ("1 Gbps", '1 Gbps'),
        ("2.4GHz", '2.4GHz'),
        ("89M users", '89M users'),
        ("5G / 4K", '5G / 4K'),
        ("$89M", '$89M'),
        ("$30", '$30'),
        ("$30.01", '$30.01'),
        ("$30.16", '$30.16'),
        ("$1.16", '$1.16'),
        ("€30.01", '€30.01'),
        ("€1.16", '€1.16'),
        ("£30.01", '£30.01'),
        ("£1.16", '£1.16'),
    ],
)
def test_units_and_money_rules(raw: str, expected: str) -> None:
    assert normalizer.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "8-12只小狗，3%-5%的小猫",
            '8-12只小狗，3%-5%的小猫',
        ),
        ("3-5天", '3-5天'),
        ("重量10-20kg", '重量10-20kg'),
        ("第3-5章", '第3-5章'),
        ("2024-2026年", '2024-2026年'),
        ("9:00-10:30", '9:00-10:30'),
        ("范围A-Z", '范围A-Z'),
        ("3%-5%", '3%-5%'),
        ("3-5%", '3-5%'),
        ("3%~5%", '3%~5%'),
        ("3％—5％", '3％—5％'),
        ("3%至5%", '3%至5%'),
        ("3.5%", '百分之3.5'),
        ("100%", '百分之百'),
        ("下降1%", '下降百分之一'),
    ],
)
def test_ranges_and_percentages_in_chinese(raw: str, expected: str) -> None:
    assert normalizer.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2-1=1", '2减1=1'),
        ("2 - 1", '2减1'),
        ("5%-3%=2%", '百分之五减百分之三=百分之二'),
        ("-5", '-5'),
        ("-3%", '-百分之三'),
        ("温度是-5°C", '温度是-5°C'),
        ("温度是-5°F", '温度是-5°F'),
    ],
)
def test_math_negatives_and_temperature(raw: str, expected: str) -> None:
    assert normalizer.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The range is 3%-5%.", 'The range is 3%-5%.'),
        ("Calculate 2-1=1.", 'Calculate 2 minus 1=1.'),
        ("temperature -5°C", 'temperature -5°C'),
        ("Use chapters 3-5 chapters", 'Use chapters 3-5 chapters'),
        ("Use A-Z", 'Use A-Z'),
        ("state-of-the-art real-time", 'state-of-the-art real-time'),
    ],
)
def test_english_hyphen_and_percent_rules(raw: str, expected: str) -> None:
    assert normalizer.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("日期是2026-09-02", '日期是2026-09-02'),
        ("编号G-123456", '编号G-123456'),
        ("G-123456", 'G-123456'),
        ("RTX-5090", 'RTX-5090'),
        ("H100-SXM", 'H100-SXM'),
        ("010-1234-5678", '010-1234-5678'),
        ("你好—世界", '你好—世界'),
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
        "tab/indent stripped"
    )


def test_none_is_empty() -> None:
    assert normalizer.normalize(None) == ""


@pytest.mark.parametrize(("raw", "expected"), [
    ("$1.23900", "$1.23900"), ("€0.001", "€0.001"),
    ("1.2300GB", "1.2300GB"), ("89M users", "89M users"),
    ("3.500%", "百分之3.500"),
    ("Rate 3.500%", "Rate 3.500 percent"),
    ("1.23900 - 0.00100", "1.23900减0.00100"),
    ("2--1=3", "2减-1=3"), ("-2-1=-3", "-2减1=-3"),
    ("算式2-1=1", "算式2减1=1"),
    ("1/2", "1/2"), ("2+1=3", "2+1=3"),
    ("2026-09-13", "2026-09-13"),
    ("3.50%-5.00%", "3.50%-5.00%"),
    ("3.50% - 1.00%=2.50%", "百分之3.50减百分之1.00=百分之2.50"),
    ("https://example.com/16GB?q=$1.239", "https://example.com/16GB?q=$1.239"),
    ("[happy] 8-12只 [pause:0.50]", "[happy] 8-12只 [pause:0.50]"),
])
def test_model_owned_numbers_and_only_subtraction(raw, expected):
    assert normalizer.normalize(raw) == expected
