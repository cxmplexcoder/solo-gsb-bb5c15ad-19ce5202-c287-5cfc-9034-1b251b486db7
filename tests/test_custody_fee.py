"""保管计费试算接口的单元测试与 API 集成测试。"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.custody_fee import CustodyFeeQuoteError, quote_custody_fee
from app.main import app
from app.models import (
    CustodianRateSchedule,
    CustodyFeeQuoteRequest,
    CustodyRateWindow,
    Event,
    EventType,
)

client = TestClient(app)

DAY = "2026-09-19"
DAY2 = "2026-09-20"


def dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime.fromisoformat(f"{DAY}T{hour:02d}:{minute:02d}:{second:02d}+00:00")


def ev(type_: EventType, hour: int, minute: int = 0, **kw) -> Event:
    return Event(type=type_, timestamp=dt(hour, minute), **kw)


def win(from_h: int, to_h: int, rate: str, from_m: int = 0, to_m: int = 0) -> CustodyRateWindow:
    # hour 可为 24（次日零点），用相对当日零点的 timedelta 构造
    base = datetime.fromisoformat(f"{DAY}T00:00:00+00:00")
    return CustodyRateWindow(
        effective_from=base + timedelta(hours=from_h, minutes=from_m),
        effective_to=base + timedelta(hours=to_h, minutes=to_m),
        hourly_rate=Decimal(rate),
    )


def schedule(custodian: str, windows) -> CustodianRateSchedule:
    return CustodianRateSchedule(custodian=custodian, windows=list(windows))


def make_request(events, settlement_hour=20, rates=()) -> CustodyFeeQuoteRequest:
    return CustodyFeeQuoteRequest(
        events=list(events),
        settlement_until=dt(settlement_hour),
        rates=list(rates),
    )


def priced_tuples(resp):
    return [
        (
            s.custodian,
            s.start.strftime("%H:%M"),
            s.end.strftime("%H:%M"),
            s.seconds,
            str(s.hourly_rate),
            str(s.amount),
            s.custodian_event_index,
        )
        for s in resp.segments
    ]


def unpriced_tuples(resp):
    return [
        (
            s.custodian,
            s.start.strftime("%H:%M"),
            s.end.strftime("%H:%M"),
            s.seconds,
            s.custodian_event_index,
        )
        for s in resp.unpriced
    ]


def simple_chain():
    """Alice 08 点封装，12 点交接给 Bob。"""
    return [
        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 12, custodian="Bob"),
    ]


# ---------------------------------------------------------------------------
# 基本计费：保管区间按费率边界切分
# ---------------------------------------------------------------------------


def test_basic_pricing_segments_split_at_rate_boundaries():
    resp = quote_custody_fee(
        make_request(
            simple_chain(),
            rates=[
                schedule("Alice", [win(0, 10, "10.5"), win(10, 24, "12")]),
                schedule("Bob", [win(12, 15, "20")]),
            ],
        )
    )
    # Alice [08,12)：[08,10) @10.5 = 21；[10,12) @12 = 24
    # Bob [12,20)：[12,15) @20 = 60；[15,20) 无费率 → unpriced
    assert priced_tuples(resp) == [
        ("Alice", "08:00", "10:00", 7200.0, "10.5", "21", 0),
        ("Alice", "10:00", "12:00", 7200.0, "12", "24", 0),
        ("Bob", "12:00", "15:00", 10800.0, "20", "60", 1),
    ]
    assert unpriced_tuples(resp) == [("Bob", "15:00", "20:00", 18000.0, 1)]
    assert [(c.custodian, str(c.subtotal)) for c in resp.custodians] == [
        ("Alice", "45.00"),
        ("Bob", "60.00"),
    ]
    assert str(resp.total) == "105.00"


def test_unpriced_when_custodian_has_no_schedule():
    # 未提供任何费率：全部时段 unpriced，小计按保管人仍列出且为 0.00
    resp = quote_custody_fee(make_request(simple_chain()))
    assert resp.segments == []
    assert unpriced_tuples(resp) == [
        ("Alice", "08:00", "12:00", 14400.0, 0),
        ("Bob", "12:00", "20:00", 28800.0, 1),
    ]
    assert [(c.custodian, str(c.subtotal)) for c in resp.custodians] == [
        ("Alice", "0.00"),
        ("Bob", "0.00"),
    ]
    assert str(resp.total) == "0.00"


def test_rate_window_partially_overlapping_interval_both_sides_gap():
    # 费率窗口 [10,11) 落在保管区间 [08,20) 中间：两侧 unpriced
    resp = quote_custody_fee(
        make_request(
            [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rates=[schedule("Alice", [win(10, 11, "1")])],
        )
    )
    assert priced_tuples(resp) == [("Alice", "10:00", "11:00", 3600.0, "1", "1", 0)]
    assert unpriced_tuples(resp) == [
        ("Alice", "08:00", "10:00", 7200.0, 0),
        ("Alice", "11:00", "20:00", 32400.0, 0),
    ]


def test_rate_windows_may_be_given_out_of_order():
    # 乱序提供的窗口按起点排序后参与切分，结果不变
    resp = quote_custody_fee(
        make_request(
            [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rates=[
                schedule(
                    "Alice",
                    [win(10, 12, "20"), win(8, 10, "10")],
                )
            ],
        )
    )
    assert priced_tuples(resp) == [
        ("Alice", "08:00", "10:00", 7200.0, "10", "20", 0),
        ("Alice", "10:00", "12:00", 7200.0, "20", "40", 0),
    ]


def test_clipped_to_settlement_until():
    # 费率窗口超出结算截止：分段只计到 settlement_until
    resp = quote_custody_fee(
        make_request(
            [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            settlement_hour=10,
            rates=[schedule("Alice", [win(0, 24, "10")])],
        )
    )
    assert priced_tuples(resp) == [("Alice", "08:00", "10:00", 7200.0, "10", "20", 0)]
    assert resp.unpriced == []
    assert str(resp.total) == "20.00"


# ---------------------------------------------------------------------------
# 精确求和与 ROUND_HALF_UP 到分；total 由已舍入小计相加
# ---------------------------------------------------------------------------


def test_subtotal_sums_exact_before_rounding():
    # 两段各为 0.005：精确求和为 0.01 后舍入；若逐段先舍入（各 0.01）会得到 0.02
    resp = quote_custody_fee(
        make_request(
            [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            settlement_hour=10,
            rates=[
                schedule(
                    "Alice",
                    [win(8, 9, "0.005"), win(9, 10, "0.005")],
                )
            ],
        )
    )
    assert [str(s.amount) for s in resp.segments] == ["0.005", "0.005"]
    assert str(resp.custodians[0].subtotal) == "0.01"
    assert str(resp.total) == "0.01"


def test_subsecond_duration_priced_exactly():
    # 保管 1 小时 1 微秒，费率 3600/小时 → 金额 = 3600 + 0.000001 × 3600/3600
    # = 3600.000001（不被截断为整数秒）
    start = datetime.fromisoformat(f"{DAY}T08:00:00+00:00")
    events = [
        Event(type=EventType.SEAL, timestamp=start, seal_id="S1", custodian="Alice"),
    ]
    request = CustodyFeeQuoteRequest(
        events=events,
        settlement_until=start + timedelta(hours=1, microseconds=1),
        rates=[
            CustodianRateSchedule(
                custodian="Alice",
                windows=[
                    CustodyRateWindow(
                        effective_from=start,
                        effective_to=start + timedelta(hours=2),
                        hourly_rate=Decimal("3600"),
                    )
                ],
            )
        ],
    )
    resp = quote_custody_fee(request)
    assert resp.segments[0].seconds == 3600.000001
    assert resp.segments[0].amount == Decimal("3600.000001")
    assert str(resp.custodians[0].subtotal) == "3600.00"


def test_round_half_up_to_cent():
    # 单段金额 0.005 → ROUND_HALF_UP 进位为 0.01；0.004 → 舍为 0.00
    for rate, expected in (("0.005", "0.01"), ("0.004", "0.00")):
        resp = quote_custody_fee(
            make_request(
                [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
                settlement_hour=9,
                rates=[schedule("Alice", [win(8, 9, rate)])],
            )
        )
        assert str(resp.custodians[0].subtotal) == expected


def test_decimal_rate_no_binary_float_error():
    # 每小时 0.1 保管 5 分钟（300 秒）：0.1 × 300/3600 = 0.008333...
    resp = quote_custody_fee(
        make_request(
            [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            settlement_hour=9,
            rates=[
                schedule(
                    "Alice",
                    [CustodyRateWindow(
                        effective_from=dt(8),
                        effective_to=dt(8, 5),
                        hourly_rate=Decimal("0.1"),
                    )],
                )
            ],
        )
    )
    assert resp.segments[0].seconds == 300.0
    # 40 位有效数字的精确十进制（1/120）
    assert resp.segments[0].amount == Decimal(
        "0.008333333333333333333333333333333333333333"
    )
    assert str(resp.custodians[0].subtotal) == "0.01"


def test_total_sums_rounded_subtotals_not_raw_amounts():
    # Alice 两段各 0.004（精确和 0.008 → 小计 0.01）；Bob 一段 0.004（小计 0.00）
    # total = 0.01 + 0.00 = 0.01，而不是未舍入总额 0.012 → 0.01（本例数值相同）；
    # 改用两段使差异显现：Alice 0.006（→0.01），Bob 0.006（→0.01）：
    # 已舍入小计之和 = 0.02；未舍入之和 0.012 直接舍入仅 0.01
    events = [
        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 9, custodian="Bob"),
    ]
    resp = quote_custody_fee(
        make_request(
            events,
            settlement_hour=10,
            rates=[
                schedule("Alice", [win(8, 9, "0.006")]),
                schedule("Bob", [win(9, 10, "0.006")]),
            ],
        )
    )
    assert [str(c.subtotal) for c in resp.custodians] == ["0.01", "0.01"]
    assert str(resp.total) == "0.02"


# ---------------------------------------------------------------------------
# 保管区间推导：unseal 不换保管人 / 违规事件不产生区间 / reseal 换保管人
# ---------------------------------------------------------------------------


def test_unseal_does_not_change_custodian():
    events = [
        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
        ev(EventType.UNSEAL, 9),
        ev(EventType.RESEAL, 10, seal_id="S2", custodian="Bob"),
    ]
    resp = quote_custody_fee(
        make_request(
            events,
            rates=[
                schedule("Alice", [win(8, 20, "1")]),
                schedule("Bob", [win(8, 20, "2")]),
            ],
        )
    )
    # Alice 持有 [08,10)（开封期间保管人不变），Bob [10,20)
    assert [(s.custodian, s.start.strftime("%H:%M"), s.end.strftime("%H:%M"))
            for s in resp.segments] == [
        ("Alice", "08:00", "10:00"),
        ("Bob", "10:00", "20:00"),
    ]
    assert resp.unpriced == []


def test_violation_events_produce_no_intervals():
    # 已封装状态下重复 seal 违规：不产生新区间，保管人仍为 Alice
    events = [
        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
        ev(EventType.SEAL, 9, seal_id="S9", custodian="Bob"),  # 违规
    ]
    resp = quote_custody_fee(
        make_request(events, rates=[schedule("Alice", [win(8, 20, "1")])])
    )
    assert resp.validation.valid is False
    assert resp.unpriced == []
    assert len(resp.segments) == 1
    assert resp.segments[0].custodian == "Alice"
    assert resp.segments[0].end == dt(20)


def test_reseal_with_same_custodian_keeps_single_priced_span():
    events = [
        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
        ev(EventType.UNSEAL, 9),
        ev(EventType.RESEAL, 10, seal_id="S2", custodian="Alice"),
    ]
    resp = quote_custody_fee(
        make_request(events, rates=[schedule("Alice", [win(8, 20, "3")])])
    )
    # 同保管人两段区间都被同一费率覆盖：[08,10) 与 [10,20)
    assert [(s.start.strftime("%H:%M"), s.end.strftime("%H:%M"), str(s.amount))
            for s in resp.segments] == [
        ("08:00", "10:00", "6"),
        ("10:00", "20:00", "30"),
    ]
    assert str(resp.custodians[0].subtotal) == "36.00"


def test_zero_length_interval_at_settlement_dropped():
    events = [
        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
        ev(EventType.HANDOVER, 10, custodian="Bob"),
    ]
    resp = quote_custody_fee(
        make_request(
            events,
            settlement_hour=10,
            rates=[schedule("Alice", [win(0, 24, "1")]), schedule("Bob", [win(0, 24, "1")])],
        )
    )
    # Bob 的末个区间退化为零长度：不出现；仅 Alice [08,10)
    assert [c.custodian for c in resp.custodians] == ["Alice"]
    assert priced_tuples(resp) == [("Alice", "08:00", "10:00", 7200.0, "1", "2", 0)]


def test_empty_events_zero_quote():
    resp = quote_custody_fee(make_request([]))
    assert resp.validation.valid is True
    assert resp.segments == []
    assert resp.unpriced == []
    assert resp.custodians == []
    assert str(resp.total) == "0.00"


def test_validation_result_preserved():
    resp = quote_custody_fee(make_request(simple_chain()))
    assert resp.validation.valid is True
    assert resp.validation.sealed is True
    assert resp.validation.final_custodian == "Bob"
    assert [e.status for e in resp.validation.events] == ["ok", "ok"]


# ---------------------------------------------------------------------------
# 422：窗口重叠 / 起止非法 / 保管人重复 / 混用时区 / 截止过早
# ---------------------------------------------------------------------------


def test_overlapping_rate_windows_raises_422():
    with pytest.raises(CustodyFeeQuoteError, match="费率窗口重叠"):
        quote_custody_fee(
            make_request(
                simple_chain(),
                rates=[
                    schedule("Alice", [win(8, 10, "1"), win(9, 11, "2")]),
                ],
            )
        )


def test_adjacent_rate_windows_allowed():
    # 首尾相接（半开）允许，且相接点两侧费率各自生效
    resp = quote_custody_fee(
        make_request(
            [ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            settlement_hour=12,
            rates=[schedule("Alice", [win(8, 10, "1"), win(10, 12, "2")])],
        )
    )
    assert priced_tuples(resp) == [
        ("Alice", "08:00", "10:00", 7200.0, "1", "2", 0),
        ("Alice", "10:00", "12:00", 7200.0, "2", "4", 0),
    ]


def test_rate_window_from_not_before_to_raises_422():
    with pytest.raises(CustodyFeeQuoteError, match="effective_from 必须早于"):
        quote_custody_fee(
            make_request(
                simple_chain(),
                rates=[schedule("Alice", [win(10, 8, "1")])],
            )
        )


def test_duplicate_custodian_rate_schedule_rejected_by_model():
    with pytest.raises(ValidationError):
        make_request(
            simple_chain(),
            rates=[schedule("Alice", [win(8, 9, "1")]), schedule("Alice", [win(9, 10, "1")])],
        )


def test_negative_rate_rejected_by_model():
    with pytest.raises(ValidationError):
        CustodyRateWindow(effective_from=dt(8), effective_to=dt(9), hourly_rate=Decimal("-0.01"))


def test_settlement_before_last_valid_event_raises_422():
    with pytest.raises(CustodyFeeQuoteError, match="settlement_until 早于最后一个有效事件"):
        quote_custody_fee(make_request(simple_chain(), settlement_hour=11))


def test_settlement_equal_to_last_valid_event_allowed():
    resp = quote_custody_fee(
        make_request(
            simple_chain(),
            settlement_hour=12,
            rates=[schedule("Alice", [win(0, 24, "1")])],
        )
    )
    # Bob 的 [12,12) 零长度丢弃；Alice [08,12) = 4
    assert str(resp.total) == "4.00"
    assert [c.custodian for c in resp.custodians] == ["Alice"]


def test_violation_event_does_not_advance_valid_baseline():
    # 8 点有效 seal，18 点违规 handover（未封装）：settlement=10 仍合法
    events = [
        ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
        ev(EventType.UNSEAL, 9),
        ev(EventType.HANDOVER, 18, custodian="Bob"),  # 未封装交接，违规
    ]
    resp = quote_custody_fee(
        make_request(events, settlement_hour=10, rates=[schedule("Alice", [win(0, 24, "1")])])
    )
    assert resp.validation.valid is False
    # unseal 不改变保管人：Alice 持有 [08,10) 共 2 小时；违规 handover 不产生区间
    assert unpriced_tuples(resp) == []
    assert str(resp.custodians[0].subtotal) == "2.00"
    assert str(resp.total) == "2.00"


def test_mixed_timezone_events_vs_settlement_raises_422():
    request = CustodyFeeQuoteRequest.model_validate(
        {
            "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                 "seal_id": "S1", "custodian": "Alice"}
            ],
            "settlement_until": f"{DAY}T20:00:00",
            "rates": [],
        }
    )
    with pytest.raises(CustodyFeeQuoteError, match="时区"):
        quote_custody_fee(request)


def test_mixed_timezone_rate_window_raises_422():
    request = CustodyFeeQuoteRequest.model_validate(
        {
            "events": [
                {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
                 "seal_id": "S1", "custodian": "Alice"}
            ],
            "settlement_until": f"{DAY}T20:00:00Z",
            "rates": [
                {
                    "custodian": "Alice",
                    "windows": [
                        {
                            "effective_from": f"{DAY}T08:00:00Z",
                            "effective_to": f"{DAY}T10:00:00",  # naive
                            "hourly_rate": "1",
                        }
                    ],
                }
            ],
        }
    )
    with pytest.raises(CustodyFeeQuoteError, match="时区"):
        quote_custody_fee(request)


def test_overlapping_windows_from_different_custodians_allowed():
    # 不同保管人的窗口可以时间重叠（各自独立）
    resp = quote_custody_fee(
        make_request(
            simple_chain(),
            rates=[
                schedule("Alice", [win(8, 20, "1")]),
                schedule("Bob", [win(8, 20, "2")]),
            ],
        )
    )
    assert resp.unpriced == []
    # Alice [08,12) 4 小时 × 1 = 4；Bob [12,20) 8 小时 × 2 = 16
    assert str(resp.total) == "20.00"


# ---------------------------------------------------------------------------
# HTTP 集成测试
# ---------------------------------------------------------------------------


def _http_payload():
    return {
        "events": [
            {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
             "seal_id": "S1", "custodian": "Alice"},
            {"type": "handover", "timestamp": f"{DAY}T12:00:00Z", "custodian": "Bob"},
        ],
        "settlement_until": f"{DAY}T20:00:00Z",
        "rates": [
            {
                "custodian": "Alice",
                "windows": [
                    {"effective_from": f"{DAY}T00:00:00Z",
                     "effective_to": f"{DAY}T10:00:00Z", "hourly_rate": "10.5"},
                    {"effective_from": f"{DAY}T10:00:00Z",
                     "effective_to": f"{DAY2}T00:00:00Z", "hourly_rate": "12.00"},
                ],
            },
            {
                "custodian": "Bob",
                "windows": [
                    {"effective_from": f"{DAY}T12:00:00Z",
                     "effective_to": f"{DAY}T15:00:00Z", "hourly_rate": "20"}
                ],
            },
        ],
    }


def test_http_endpoint_success():
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=_http_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["settlement_until"] == f"{DAY}T20:00:00Z"
    assert data["segments"] == [
        {"custodian": "Alice", "start": f"{DAY}T08:00:00Z", "end": f"{DAY}T10:00:00Z",
         "seconds": 7200.0, "hourly_rate": "10.5", "amount": "21",
         "custodian_event_index": 0},
        {"custodian": "Alice", "start": f"{DAY}T10:00:00Z", "end": f"{DAY}T12:00:00Z",
         "seconds": 7200.0, "hourly_rate": "12.00", "amount": "24",
         "custodian_event_index": 0},
        {"custodian": "Bob", "start": f"{DAY}T12:00:00Z", "end": f"{DAY}T15:00:00Z",
         "seconds": 10800.0, "hourly_rate": "20", "amount": "60",
         "custodian_event_index": 1},
    ]
    assert data["unpriced"] == [
        {"custodian": "Bob", "start": f"{DAY}T15:00:00Z", "end": f"{DAY}T20:00:00Z",
         "seconds": 18000.0, "custodian_event_index": 1}
    ]
    assert data["custodians"] == [
        {"custodian": "Alice", "subtotal": "45.00"},
        {"custodian": "Bob", "subtotal": "60.00"},
    ]
    assert data["total"] == "105.00"


def test_http_amounts_serialize_as_decimal_strings():
    payload = _http_payload()
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    for segment in response.json()["segments"]:
        assert isinstance(segment["hourly_rate"], str)
        assert isinstance(segment["amount"], str)
    assert isinstance(response.json()["total"], str)


def test_http_overlapping_windows_returns_422():
    payload = _http_payload()
    payload["rates"][0]["windows"].append(
        {"effective_from": f"{DAY}T09:00:00Z",
         "effective_to": f"{DAY}T11:00:00Z", "hourly_rate": "99"}
    )
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422
    assert "费率窗口重叠" in response.json()["detail"]


def test_http_settlement_too_early_returns_422():
    payload = _http_payload()
    payload["settlement_until"] = f"{DAY}T11:00:00Z"
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422
    assert "settlement_until" in response.json()["detail"]


def test_http_mixed_timezone_returns_422():
    payload = _http_payload()
    payload["settlement_until"] = f"{DAY}T20:00:00"  # naive
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422
    assert "时区" in response.json()["detail"]


def test_http_duplicate_custodian_rate_returns_422():
    payload = _http_payload()
    payload["rates"].append({"custodian": "Alice", "windows": []})
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422


def test_http_invalid_event_structure_returns_422():
    payload = _http_payload()
    payload["events"][0]["type"] = "nonsense"
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422
