"""保管计费试算。

处理流程：
1. 调用现有单链校验器（``validate_events``）得到原校验结果并原样保留；
2. 仅以其中 ``status=ok`` 的 seal / reseal / handover 事件生成保管人半开区间
   ``[start, end)``：每个有效事件以其时间戳开启对应保管人的区间，下一个有效
   事件闭合前一区间，末个区间延伸至 ``settlement_until``；unseal 不改变保管人，
   违规事件一律不产生区间，与 ``settlement_until`` 重合的零长度区间丢弃；
3. 请求按保管人提供互不重叠的生效费率窗口（半开 ``[start, end)``，允许乱序与
   首尾相接）与每小时十进制费率；保管人费率计划重复、同一保管人的费率窗口
   重叠均返回 422；
4. 将每段保管区间按费率边界切分（复用 ``timeline.split_by_cover`` 半开区间
   分割器）：被费率窗口覆盖的分段逐段返回秒数、适用费率与未舍入金额
   （``hourly_rate × seconds / 3600``，十进制精确计算），未被任何窗口覆盖的
   时段列入 ``unpriced``；
5. 每位保管人的分段金额先精确求和，再按 ROUND_HALF_UP 舍入到分（0.01）；
   ``total`` 由已舍入小计相加得到。

事件、费率窗口与 ``settlement_until`` 的时间不得混用带时区与不带时区的时间
（否则 422）；``settlement_until`` 早于最后一个有效事件时间时返回 422。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

from .models import (
    CustodianFeeSummary,
    CustodyFeeQuoteRequest,
    CustodyFeeQuoteResponse,
    EventType,
    FeeRateWindow,
    FeeSegment,
    UnpricedSegment,
    ValidateResponse,
)
from .timeline import (
    CoverInterval,
    ensure_comparable,
    ok_event_records,
    split_by_cover,
)
from .validator import validate_events


class CustodyFeeQuoteError(ValueError):
    """试算入参的语义错误（在路由层转换为 HTTP 422）。"""


# 生成保管区间的事件类型：seal / reseal / handover（unseal 不改变保管人）
_CUSTODIAN_TYPES = (EventType.SEAL, EventType.RESEAL, EventType.HANDOVER)

#: 小计舍入单位：分（0.01），ROUND_HALF_UP
_CENT = Decimal("0.01")
#: 每小时秒数（十进制）
_SECONDS_PER_HOUR = Decimal(3600)


@dataclass(frozen=True)
class _CustodyInterval:
    """一段保管人半开区间 [start, end)。"""

    custodian: str
    start: datetime
    end: datetime


def _exact_seconds(start: datetime, end: datetime) -> Decimal:
    """半开区间时长的精确十进制秒数。

    timedelta 内部以整数天 / 秒 / 微秒存储，可无损转换为十进制。
    """
    delta = end - start
    return (
        Decimal(delta.days * 86400 + delta.seconds)
        + Decimal(delta.microseconds) / Decimal(1_000_000)
    )


def _ensure_non_overlapping(
    custodian: str, windows: List[FeeRateWindow]
) -> List[FeeRateWindow]:
    """按起点排序并确认同一保管人的费率窗口互不重叠，返回排序后的窗口。

    半开区间下首尾相接（前一窗口 end == 后一窗口 start）不算重叠。
    """
    ordered = sorted(windows, key=lambda window: (window.start, window.end))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.end:
            raise CustodyFeeQuoteError(
                f"保管人 {custodian} 的费率窗口重叠："
                f"[{previous.start.isoformat()}, {previous.end.isoformat()}) 与 "
                f"[{current.start.isoformat()}, {current.end.isoformat()})"
            )
    return ordered


def _derive_custody_intervals(
    request: CustodyFeeQuoteRequest, validation: ValidateResponse
) -> List[_CustodyInterval]:
    """从 status=ok 的 seal/reseal/handover 事件生成保管人半开区间。

    每个有效事件开启对应保管人的区间，下一个有效事件闭合前一区间；
    末个区间延伸至 settlement_until；与 settlement_until 重合的零长度
    区间丢弃。区间按时间升序返回。
    """
    records = ok_event_records(request.events, validation.events, _CUSTODIAN_TYPES)
    intervals: List[_CustodyInterval] = []
    for i, record in enumerate(records):
        end = records[i + 1].timestamp if i + 1 < len(records) else request.settlement_until
        if record.timestamp < end:
            intervals.append(
                _CustodyInterval(
                    custodian=record.custodian,  # type: ignore[arg-type]
                    start=record.timestamp,
                    end=end,
                )
            )
    return intervals


def quote_custody_fee(request: CustodyFeeQuoteRequest) -> CustodyFeeQuoteResponse:
    """校验事件链并按保管人费率窗口对保管区间计费试算。"""
    # 阶段 1：复用单链校验，结果原样保留
    validation = validate_events(request.events)

    # 阶段 2：事件、费率窗口与 settlement_until 不得混用时区
    timestamps = [request.settlement_until]
    timestamps.extend(event.timestamp for event in request.events)
    for plan in request.rate_plans:
        for window in plan.windows:
            timestamps.append(window.start)
            timestamps.append(window.end)
    ensure_comparable(timestamps, CustodyFeeQuoteError)

    # 阶段 3：保管人费率计划不可重复；同一保管人的费率窗口互不重叠
    plans: Dict[str, List[FeeRateWindow]] = {}
    for plan in request.rate_plans:
        if plan.custodian in plans:
            raise CustodyFeeQuoteError(f"保管人费率计划重复：{plan.custodian}")
        plans[plan.custodian] = _ensure_non_overlapping(plan.custodian, plan.windows)

    # 阶段 4：settlement_until 不得早于最后一个有效事件时间（违规事件不推进基准）
    last_valid: Optional[datetime] = None
    for event, event_result in zip(request.events, validation.events):
        if event_result.status != "ok":
            continue
        if last_valid is None or event.timestamp > last_valid:
            last_valid = event.timestamp
    if last_valid is not None and request.settlement_until < last_valid:
        raise CustodyFeeQuoteError(
            "settlement_until 早于最后一个有效事件时间："
            f"{request.settlement_until.isoformat()} 早于 {last_valid.isoformat()}"
        )

    # 阶段 5：生成保管区间（时间升序）
    intervals = _derive_custody_intervals(request, validation)

    # 阶段 6：按费率边界切分各保管区间，逐段计价或列入 unpriced
    segments: List[FeeSegment] = []
    unpriced: List[UnpricedSegment] = []
    exact_totals: Dict[str, Decimal] = {}  # 按首次持有时间升序（dict 保序）
    for interval in intervals:
        exact_totals.setdefault(interval.custodian, Decimal(0))
        covers = [
            CoverInterval(start=window.start, end=window.end, payload=window)
            for window in plans.get(interval.custodian, [])
        ]
        for piece in split_by_cover(interval.start, interval.end, covers):
            seconds = _exact_seconds(piece.start, piece.end)
            if piece.covered:
                window = piece.payload
                assert window is not None  # covered 片段必携带费率窗口
                amount = window.hourly_rate * seconds / _SECONDS_PER_HOUR
                segments.append(
                    FeeSegment(
                        custodian=interval.custodian,
                        start=piece.start,
                        end=piece.end,
                        seconds=(piece.end - piece.start).total_seconds(),
                        hourly_rate=window.hourly_rate,
                        amount=amount,
                    )
                )
                exact_totals[interval.custodian] += amount
            else:
                unpriced.append(
                    UnpricedSegment(
                        custodian=interval.custodian,
                        start=piece.start,
                        end=piece.end,
                        seconds=(piece.end - piece.start).total_seconds(),
                    )
                )

    # 阶段 7：每位保管人先精确求和再按 ROUND_HALF_UP 舍入到分；total 由小计相加
    custodians = [
        CustodianFeeSummary(
            custodian=custodian,
            amount=exact.quantize(_CENT, rounding=ROUND_HALF_UP),
        )
        for custodian, exact in exact_totals.items()
    ]
    total = sum((summary.amount for summary in custodians), Decimal("0.00"))

    return CustodyFeeQuoteResponse(
        settlement_until=request.settlement_until,
        validation=validation,
        segments=segments,
        unpriced=unpriced,
        custodians=custodians,
        total=total,
    )
