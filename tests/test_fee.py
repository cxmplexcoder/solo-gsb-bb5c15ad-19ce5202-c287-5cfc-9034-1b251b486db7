"""保管计费试算接口的单元测试与 API 集成测试。"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.fee import CustodyFeeQuoteError, quote_custody_fee
from app.main import app
from app.models import (
    CustodianRatePlan,
    CustodyFeeQuoteRequest,
    Event,
    EventType,
    FeeRateWindow,
)

client = TestClient(app)

DAY = "2026-09-19"


def dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    base = datetime.fromisoformat(f"{DAY}T00:00:00+00:00")
    return base + timedelta(hours=hour, minutes=minute, seconds=second)


def ev(type_: EventType, hour: int, minute: int = 0, **kw) -> Event:
    return Event(type=type_, timestamp=dt(hour, minute), **kw)


def window(start_h: int, end_h: int, rate, start_m: int = 0, end_m: int = 0) -> FeeRateWindow:
    return FeeRateWindow(
        start=dt(start_h, start_m),
        end=dt(end_h, end_m),
        hourly_rate=Decimal(str(rate)),
    )


def plan(custodian: str, windows) -> CustodianRatePlan:
    return CustodianRatePlan(custodian=custodian, windows=list(windows))


def make_request(events, rate_plans, settlement_until=None) -> CustodyFeeQuoteRequest:
    return CustodyFeeQuoteRequest(
        events=list(events),
        rate_plans=list(rate_plans),
        settlement_until=settlement_until if settlement_until is not None else dt(20),
    )


def segments_of(resp):
    return [
        (s.custodian, s.start, s.end, s.seconds, s.hourly_rate, s.amount)
        for s in resp.segments
    ]


def unpriced_of(resp):
    return [(s.custodian, s.start, s.end, s.seconds) for s in resp.unpriced]


def summary_map(resp):
    return {c.custodian: c.amount for c in resp.custodians}


# ---------------------------------------------------------------------------
# 基本计价与费率边界切分
# ---------------------------------------------------------------------------


def test_basic_single_segment():
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(0, 24, "10")])],
            settlement_until=dt(12),
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(8), dt(12), 4 * 3600.0, Decimal("10"), Decimal("40")),
    ]
    assert resp.unpriced == []
    assert summary_map(resp) == {"Alice": Decimal("40.00")}
    assert resp.total == Decimal("40.00")


def test_split_at_rate_boundary():
    # 保管区间 [08,12) 在费率边界 10 点被切开：2h×10 + 2h×20 = 60
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(6, 10, "10"), window(10, 14, "20")])],
            settlement_until=dt(12),
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(8), dt(10), 7200.0, Decimal("10"), Decimal("20")),
        ("Alice", dt(10), dt(12), 7200.0, Decimal("20"), Decimal("40")),
    ]
    assert summary_map(resp) == {"Alice": Decimal("60.00")}
    assert resp.total == Decimal("60.00")


def test_touching_windows_are_not_overlap():
    # 半开区间下首尾相接（end == start）允许，不视为重叠
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(8, 10, "1"), window(10, 12, "2")])],
            settlement_until=dt(12),
        )
    )
    assert [s.hourly_rate for s in resp.segments] == [Decimal("1"), Decimal("2")]


def test_windows_may_be_unsorted():
    # 费率窗口允许乱序，按起点排序后参与切分
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(10, 14, "20"), window(6, 10, "10")])],
            settlement_until=dt(12),
        )
    )
    assert [s.hourly_rate for s in resp.segments] == [Decimal("10"), Decimal("20")]


def test_sub_hour_precision():
    # 90 秒 @ 36/小时 = 0.9，十进制精确计算
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(8, 9, "36")])],
            settlement_until=dt(8, 1, 30),
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(8), dt(8, 1, 30), 90.0, Decimal("36"), Decimal("0.900")),
    ]
    assert summary_map(resp) == {"Alice": Decimal("0.90")}


# ---------------------------------------------------------------------------
# unpriced：无覆盖费率的时段
# ---------------------------------------------------------------------------


def test_uncovered_period_listed_in_unpriced():
    # 费率窗口只覆盖 [06,09)：保管区间 [08,12) 中 [09,12) 无费率
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(6, 9, "10")])],
            settlement_until=dt(12),
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(8), dt(9), 3600.0, Decimal("10"), Decimal("10")),
    ]
    assert unpriced_of(resp) == [("Alice", dt(9), dt(12), 3 * 3600.0)]
    assert summary_map(resp) == {"Alice": Decimal("10.00")}
    assert resp.total == Decimal("10.00")


def test_gap_between_windows_unpriced():
    # 两个费率窗口之间的空洞列入 unpriced，分段按时间升序
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(8, 9, "10"), window(11, 12, "10")])],
            settlement_until=dt(12),
        )
    )
    assert [(s.start, s.end) for s in resp.segments] == [(dt(8), dt(9)), (dt(11), dt(12))]
    assert unpriced_of(resp) == [("Alice", dt(9), dt(11), 7200.0)]
    assert summary_map(resp) == {"Alice": Decimal("20.00")}


def test_custodian_without_plan_all_unpriced():
    # Bob 没有费率计划：全部时段 unpriced，小计 0.00 仍出现在 custodians 中
    resp = quote_custody_fee(
        make_request(
            events=[
                ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                ev(EventType.HANDOVER, 10, custodian="Bob"),
            ],
            rate_plans=[plan("Alice", [window(0, 24, "10")])],
            settlement_until=dt(12),
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(8), dt(10), 7200.0, Decimal("10"), Decimal("20")),
    ]
    assert unpriced_of(resp) == [("Bob", dt(10), dt(12), 7200.0)]
    assert summary_map(resp) == {"Alice": Decimal("20.00"), "Bob": Decimal("0.00")}
    assert resp.total == Decimal("20.00")


def test_plan_for_inactive_custodian_ignored():
    # 费率计划中的保管人未实际持有展品：不出现在结果中
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(0, 24, "10")]), plan("Zoe", [window(0, 24, "5")])],
            settlement_until=dt(12),
        )
    )
    assert set(summary_map(resp)) == {"Alice"}


# ---------------------------------------------------------------------------
# 多保管人与舍入规则
# ---------------------------------------------------------------------------


def test_handover_splits_by_custodian():
    # Alice [08,10) @12 → 24；Bob [10,12) @18 → 36；小计分列，total 相加
    resp = quote_custody_fee(
        make_request(
            events=[
                ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                ev(EventType.HANDOVER, 10, custodian="Bob"),
            ],
            rate_plans=[
                plan("Alice", [window(0, 24, "12")]),
                plan("Bob", [window(0, 24, "18")]),
            ],
            settlement_until=dt(12),
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(8), dt(10), 7200.0, Decimal("12"), Decimal("24")),
        ("Bob", dt(10), dt(12), 7200.0, Decimal("18"), Decimal("36")),
    ]
    assert [c.custodian for c in resp.custodians] == ["Alice", "Bob"]
    assert summary_map(resp) == {"Alice": Decimal("24.00"), "Bob": Decimal("36.00")}
    assert resp.total == Decimal("60.00")


def test_exact_sum_before_rounding():
    # 两段各 0.005：先精确求和得 0.010 再舍入为 0.01（逐段舍入会错误得到 0.02）
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[
                plan("Alice", [window(8, 9, "0.005"), window(9, 10, "0.005")])
            ],
            settlement_until=dt(10),
        )
    )
    assert [s.amount for s in resp.segments] == [Decimal("0.005"), Decimal("0.005")]
    assert summary_map(resp) == {"Alice": Decimal("0.01")}
    assert resp.total == Decimal("0.01")


def test_round_half_up_not_bankers():
    # 0.005 按 ROUND_HALF_UP 进位为 0.01（而非银行家舍入的 0.00）
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(8, 9, "0.005")])],
            settlement_until=dt(9),
        )
    )
    assert summary_map(resp) == {"Alice": Decimal("0.01")}


def test_total_is_sum_of_rounded_subtotals():
    # Alice / Bob 各 0.005 → 各自舍入为 0.01；total = 0.01 + 0.01 = 0.02
    # （若对精确总额 0.010 舍入只会得到 0.01）
    resp = quote_custody_fee(
        make_request(
            events=[
                ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                ev(EventType.HANDOVER, 9, custodian="Bob"),
            ],
            rate_plans=[
                plan("Alice", [window(8, 9, "0.005")]),
                plan("Bob", [window(9, 10, "0.005")]),
            ],
            settlement_until=dt(10),
        )
    )
    assert summary_map(resp) == {"Alice": Decimal("0.01"), "Bob": Decimal("0.01")}
    assert resp.total == Decimal("0.02")


def test_custodians_ordered_by_first_holding():
    # 保管人小计按首次持有时间升序（Bob 先持有）
    resp = quote_custody_fee(
        make_request(
            events=[
                ev(EventType.SEAL, 8, seal_id="S1", custodian="Bob"),
                ev(EventType.HANDOVER, 10, custodian="Alice"),
            ],
            rate_plans=[
                plan("Alice", [window(0, 24, "1")]),
                plan("Bob", [window(0, 24, "1")]),
            ],
            settlement_until=dt(12),
        )
    )
    assert [c.custodian for c in resp.custodians] == ["Bob", "Alice"]


# ---------------------------------------------------------------------------
# 保管区间推导：违规事件 / unseal / 末段延伸 / settlement_until 截断
# ---------------------------------------------------------------------------


def test_violation_events_produce_no_intervals():
    # 已封装状态下的第二次 seal 违规，不产生新区间；链违规不影响 200
    resp = quote_custody_fee(
        make_request(
            events=[
                ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                ev(EventType.SEAL, 9, seal_id="S2", custodian="Bob"),
            ],
            rate_plans=[plan("Alice", [window(0, 24, "10")])],
            settlement_until=dt(12),
        )
    )
    assert resp.validation.valid is False
    assert segments_of(resp) == [
        ("Alice", dt(8), dt(12), 4 * 3600.0, Decimal("10"), Decimal("40")),
    ]
    assert set(summary_map(resp)) == {"Alice"}


def test_unseal_does_not_change_custodian():
    # 开封不改变保管人：reseal 在 10 点闭合前段并开启新段，两段同属 Alice
    resp = quote_custody_fee(
        make_request(
            events=[
                ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                ev(EventType.UNSEAL, 9),
                ev(EventType.RESEAL, 10, seal_id="S2", custodian="Alice"),
            ],
            rate_plans=[plan("Alice", [window(0, 24, "10")])],
            settlement_until=dt(12),
        )
    )
    assert [(s.custodian, s.start, s.end) for s in resp.segments] == [
        ("Alice", dt(8), dt(10)),
        ("Alice", dt(10), dt(12)),
    ]
    assert summary_map(resp) == {"Alice": Decimal("40.00")}


def test_settlement_until_clips_last_interval():
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(0, 24, "10")])],
            settlement_until=dt(10),
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(8), dt(10), 7200.0, Decimal("10"), Decimal("20")),
    ]


def test_zero_length_interval_at_settlement_until_dropped():
    # 最后一个有效事件恰在 settlement_until：末个区间退化为零长度并丢弃
    resp = quote_custody_fee(
        make_request(
            events=[
                ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                ev(EventType.HANDOVER, 10, custodian="Bob"),
            ],
            rate_plans=[plan("Alice", [window(0, 24, "10")]), plan("Bob", [window(0, 24, "5")])],
            settlement_until=dt(10),
        )
    )
    assert segments_of(resp) == [
        ("Alice", dt(8), dt(10), 7200.0, Decimal("10"), Decimal("20")),
    ]
    assert set(summary_map(resp)) == {"Alice"}


def test_settlement_until_equal_to_last_valid_event_allowed():
    # 恰好相等允许；唯一的保管区间退化为零长度，不产生任何分段
    resp = quote_custody_fee(
        make_request(
            events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
            rate_plans=[plan("Alice", [window(0, 24, "10")])],
            settlement_until=dt(8),
        )
    )
    assert resp.segments == []
    assert resp.unpriced == []
    assert resp.custodians == []
    assert resp.total == Decimal("0.00")


def test_empty_events_and_plans():
    resp = quote_custody_fee(make_request(events=[], rate_plans=[]))
    assert resp.validation.valid is True
    assert resp.segments == []
    assert resp.unpriced == []
    assert resp.custodians == []
    assert resp.total == Decimal("0.00")


def test_validation_result_preserved():
    resp = quote_custody_fee(
        make_request(
            events=[
                ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                ev(EventType.HANDOVER, 9, custodian="Bob"),
            ],
            rate_plans=[],
            settlement_until=dt(12),
        )
    )
    assert resp.validation.valid is True
    assert resp.validation.final_custodian == "Bob"
    assert [e.status for e in resp.validation.events] == ["ok", "ok"]
    # 无费率计划：全部时段 unpriced
    assert unpriced_of(resp) == [
        ("Alice", dt(8), dt(9), 3600.0),
        ("Bob", dt(9), dt(12), 3 * 3600.0),
    ]
    assert summary_map(resp) == {"Alice": Decimal("0.00"), "Bob": Decimal("0.00")}


# ---------------------------------------------------------------------------
# 422：settlement_until 过早 / 混用时区 / 窗口重叠 / 计划重复
# ---------------------------------------------------------------------------


def test_settlement_until_before_last_valid_event_raises_422():
    with pytest.raises(CustodyFeeQuoteError, match="settlement_until 早于最后一个有效事件"):
        quote_custody_fee(
            make_request(
                events=[ev(EventType.SEAL, 18, seal_id="S1", custodian="Alice")],
                rate_plans=[plan("Alice", [window(0, 24, "10")])],
                settlement_until=dt(10),
            )
        )


def test_violation_event_does_not_advance_baseline():
    # 8 点有效 seal，18 点违规 handover（未封装状态下不得交接）：
    # 违规事件不推进有效时间基准，settlement_until=10 仍合法；
    # unseal 不改变保管人，Alice 的保管区间延伸至 settlement_until
    resp = quote_custody_fee(
        make_request(
            events=[
                ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                ev(EventType.UNSEAL, 9),
                ev(EventType.HANDOVER, 18, custodian="Bob"),  # 未封装交接，违规
            ],
            rate_plans=[plan("Alice", [window(0, 24, "10")])],
            settlement_until=dt(10),
        )
    )
    assert resp.validation.valid is False
    assert segments_of(resp) == [
        ("Alice", dt(8), dt(10), 7200.0, Decimal("10"), Decimal("20")),
    ]


def test_mixed_timezone_events_and_settlement_raises_422():
    request = CustodyFeeQuoteRequest.model_validate(
        {
            "events": [
                {
                    "type": "seal",
                    "timestamp": f"{DAY}T08:00:00Z",
                    "seal_id": "S1",
                    "custodian": "Alice",
                }
            ],
            "settlement_until": f"{DAY}T12:00:00",  # naive
            "rate_plans": [],
        }
    )
    with pytest.raises(CustodyFeeQuoteError, match="时区"):
        quote_custody_fee(request)


def test_mixed_timezone_window_raises_422():
    request = CustodyFeeQuoteRequest.model_validate(
        {
            "events": [
                {
                    "type": "seal",
                    "timestamp": f"{DAY}T08:00:00Z",
                    "seal_id": "S1",
                    "custodian": "Alice",
                }
            ],
            "settlement_until": f"{DAY}T12:00:00Z",
            "rate_plans": [
                {
                    "custodian": "Alice",
                    "windows": [
                        {
                            "start": f"{DAY}T00:00:00",  # naive
                            "end": f"{DAY}T23:59:00",
                            "hourly_rate": "10",
                        }
                    ],
                }
            ],
        }
    )
    with pytest.raises(CustodyFeeQuoteError, match="时区"):
        quote_custody_fee(request)


def test_overlapping_windows_raise_422():
    with pytest.raises(CustodyFeeQuoteError, match="费率窗口重叠"):
        quote_custody_fee(
            make_request(
                events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
                rate_plans=[plan("Alice", [window(6, 11, "10"), window(10, 14, "20")])],
            )
        )


def test_nested_windows_raise_422():
    with pytest.raises(CustodyFeeQuoteError, match="费率窗口重叠"):
        quote_custody_fee(
            make_request(
                events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
                rate_plans=[plan("Alice", [window(6, 14, "10"), window(8, 10, "20")])],
            )
        )


def test_overlap_checked_per_custodian():
    # 不同保管人的窗口即使相交也不冲突（各自作用于自己的保管区间）
    resp = quote_custody_fee(
        make_request(
            events=[
                ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice"),
                ev(EventType.HANDOVER, 10, custodian="Bob"),
            ],
            rate_plans=[
                plan("Alice", [window(6, 12, "10")]),
                plan("Bob", [window(8, 14, "20")]),
            ],
            settlement_until=dt(12),
        )
    )
    assert summary_map(resp) == {"Alice": Decimal("20.00"), "Bob": Decimal("40.00")}


def test_duplicate_rate_plan_raises_422():
    with pytest.raises(CustodyFeeQuoteError, match="保管人费率计划重复"):
        quote_custody_fee(
            make_request(
                events=[ev(EventType.SEAL, 8, seal_id="S1", custodian="Alice")],
                rate_plans=[plan("Alice", []), plan("Alice", [window(0, 24, "10")])],
            )
        )


# ---------------------------------------------------------------------------
# 模型层校验（pydantic → 422 结构错误）
# ---------------------------------------------------------------------------


def test_window_start_not_before_end_rejected_by_model():
    from pydantic import ValidationError

    for start_h, end_h in ((10, 10), (11, 10)):
        with pytest.raises(ValidationError):
            window(start_h, end_h, "10")


def test_window_mixed_timezone_rejected_by_model():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FeeRateWindow.model_validate(
            {
                "start": f"{DAY}T08:00:00Z",
                "end": f"{DAY}T10:00:00",
                "hourly_rate": "10",
            }
        )


def test_non_finite_rate_rejected_by_model():
    from pydantic import ValidationError

    for bad in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValidationError):
            FeeRateWindow(start=dt(8), end=dt(10), hourly_rate=Decimal(bad))


def test_empty_custodian_rejected_by_model():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        plan("", [])


# ---------------------------------------------------------------------------
# HTTP 集成测试
# ---------------------------------------------------------------------------


def _http_payload():
    return {
        "events": [
            {"type": "seal", "timestamp": f"{DAY}T08:00:00Z",
             "seal_id": "S001", "custodian": "Alice"},
            {"type": "handover", "timestamp": f"{DAY}T10:00:00Z",
             "custodian": "Bob"},
        ],
        "settlement_until": f"{DAY}T12:00:00Z",
        "rate_plans": [
            {
                "custodian": "Alice",
                "windows": [
                    {"start": f"{DAY}T06:00:00Z", "end": f"{DAY}T09:00:00Z",
                     "hourly_rate": "12.50"},
                    {"start": f"{DAY}T09:00:00Z", "end": f"{DAY}T18:00:00Z",
                     "hourly_rate": "15"},
                ],
            },
        ],
    }


def test_http_endpoint_success():
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=_http_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["settlement_until"] == f"{DAY}T12:00:00Z"
    assert data["validation"]["valid"] is True
    assert data["validation"]["final_custodian"] == "Bob"
    # Alice [08,10) 在费率边界 09 点被切开；Bob [10,12) 无费率计划
    assert data["segments"] == [
        {
            "custodian": "Alice",
            "start": f"{DAY}T08:00:00Z",
            "end": f"{DAY}T09:00:00Z",
            "seconds": 3600.0,
            "hourly_rate": "12.50",
            "amount": "12.50",
        },
        {
            "custodian": "Alice",
            "start": f"{DAY}T09:00:00Z",
            "end": f"{DAY}T10:00:00Z",
            "seconds": 3600.0,
            "hourly_rate": "15",
            "amount": "15",
        },
    ]
    assert data["unpriced"] == [
        {
            "custodian": "Bob",
            "start": f"{DAY}T10:00:00Z",
            "end": f"{DAY}T12:00:00Z",
            "seconds": 7200.0,
        }
    ]
    assert data["custodians"] == [
        {"custodian": "Alice", "amount": "27.50"},
        {"custodian": "Bob", "amount": "0.00"},
    ]
    assert data["total"] == "27.50"


def test_http_overlapping_windows_returns_422():
    payload = _http_payload()
    payload["rate_plans"][0]["windows"][1]["start"] = f"{DAY}T08:30:00Z"
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422
    assert "费率窗口重叠" in response.json()["detail"]


def test_http_duplicate_rate_plan_returns_422():
    payload = _http_payload()
    payload["rate_plans"].append({"custodian": "Alice", "windows": []})
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422
    assert "保管人费率计划重复" in response.json()["detail"]


def test_http_mixed_timezone_returns_422():
    payload = _http_payload()
    payload["settlement_until"] = f"{DAY}T12:00:00"  # naive
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422
    assert "时区" in response.json()["detail"]


def test_http_settlement_until_before_valid_event_returns_422():
    payload = _http_payload()
    payload["settlement_until"] = f"{DAY}T07:00:00Z"
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422
    assert "settlement_until" in response.json()["detail"]


def test_http_window_start_after_end_returns_422():
    payload = _http_payload()
    payload["rate_plans"][0]["windows"][0]["end"] = f"{DAY}T05:00:00Z"
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422


def test_http_non_finite_rate_returns_422():
    payload = _http_payload()
    payload["rate_plans"][0]["windows"][0]["hourly_rate"] = "Infinity"
    response = client.post("/api/v1/seal-chain/custody-fee-quote", json=payload)
    assert response.status_code == 422
