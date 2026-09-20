"""保管计费试算。

处理流程：
1. 调用现有单链校验器（``validate_events``）得到原校验结果并原样保留；
2. 仅以其中 ``status=ok`` 的 seal / reseal / handover 事件生成截至
   ``settlement_until`` 的保管人半开区间 ``[start, end)``：每个有效事件以其
   时间戳开启对应保管人的区间，下一个有效事件闭合前一区间，末个区间向右
   延伸至 ``settlement_until``；unseal 不改变保管人，违规事件一律不产生
   区间，与 ``settlement_until`` 重合的零长度区间丢弃；
3. 各保管人提供的生效费率窗口须互不重叠（允许首尾相接），可乱序提供；
   将每个保管区间按该保管人的费率边界切分，命中窗口的片段逐段返回秒数、
   适用费率与**未舍入**金额（``seconds / 3600 × hourly_rate``），无任何
   费率窗口覆盖的片段列入 ``unpriced``；
4. 每位保管人的分段金额以精确有理数（``Fraction``）求和后，再按
   ``ROUND_HALF_UP`` 舍入到分；``total`` 由各保管人**已舍入小计**直接
   相加，不再二次舍入。

事件、``settlement_until`` 与费率窗口的时间不得混用带时区与不带时区的
时间（否则 422）；费率窗口起止非法（``effective_from >= effective_to``）、
同一保管人的费率窗口重叠、保管人费率重复、或 ``settlement_until`` 早于
最后一个有效事件时间时同样返回 422。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .models import (
    CustodianFeeQuote,
    CustodianRateSchedule,
    CustodyFeeQuoteRequest,
    CustodyFeeQuoteResponse,
    Event,
    EventResult,
    EventType,
    PricedCustodySegment,
    UnpricedCustodySegment,
)
from .timeline import CoverInterval, ensure_comparable, ok_event_records, split_by_cover
from .validator import validate_events


class CustodyFeeQuoteError(ValueError):
    """计费试算入参的语义错误（在路由层转换为 HTTP 422）。"""


# 生成保管区间的事件类型：seal / reseal / handover（unseal 不改变保管人）
_CUSTODIAN_TYPES = (EventType.SEAL, EventType.RESEAL, EventType.HANDOVER)

#: 金额舍入粒度：分（0.01）
_CENT = Decimal("0.01")

#: 未舍入金额由精确有理数转十进制时的有效位数（足以覆盖任意实际费率 / 秒数）
_DECIMAL_PRECISION = 40

_SECONDS_PER_HOUR = Fraction(3600)


@dataclass(frozen=True)
class _CustodyInterval:
    """一段保管人半开区间 [start, end) 及其来源事件下标。"""

    start: datetime
    end: datetime
    custodian: str
    event_index: int


@dataclass(frozen=True)
class _Rate:
    """排序后的生效费率窗口 [start, end)；end 为 None 表示向右开放。"""

    start: datetime
    end: Optional[datetime]
    hourly_rate: Decimal


def _derive_custody_intervals(
    events: List[Event], event_results: List[EventResult], settlement_until: datetime
) -> List[_CustodyInterval]:
    """从 status=ok 的 seal/reseal/handover 事件生成截至结算点的半开保管区间。

    每个有效事件开启对应保管人的区间，下一个有效事件闭合前一区间；
    末个区间向右延伸至 settlement_until；与之重合的零长度区间丢弃。
    """
    records = ok_event_records(events, event_results, _CUSTODIAN_TYPES)
    intervals: List[_CustodyInterval] = []
    for i, record in enumerate(records):
        end = records[i + 1].timestamp if i + 1 < len(records) else settlement_until
        if record.timestamp < end:
            intervals.append(
                _CustodyInterval(
                    start=record.timestamp,
                    end=end,
                    custodian=record.custodian,  # type: ignore[arg-type]
                    event_index=record.index,
                )
            )
    return intervals


def _validate_windows(schedule: CustodianRateSchedule) -> List[_Rate]:
    """校验单个保管人的费率窗口并按起点排序。

    窗口须 effective_from 严格早于 effective_to；窗口间不得重叠
    （首尾相接允许，半开语义下相接点两侧费率各自生效）。
    """
    ordered: List[Tuple[datetime, Optional[datetime], Decimal]] = []
    for window in schedule.windows:
        try:
            start_before_end = window.effective_from < window.effective_to
        except TypeError:
            raise CustodyFeeQuoteError(
                "时间戳时区信息不一致（不能混用带时区与不带时区的时间）"
            )
        if not start_before_end:
            raise CustodyFeeQuoteError(
                f"保管人 {schedule.custodian} 的费率窗口 effective_from 必须早于 "
                f"effective_to：{window.effective_from.isoformat()} 不早于 "
                f"{window.effective_to.isoformat()}"
            )
        ordered.append(
            (window.effective_from, window.effective_to, window.hourly_rate)
        )

    ordered.sort(key=lambda item: item[0])
    rates: List[_Rate] = []
    for start, end, hourly_rate in ordered:
        if rates:
            previous_end = rates[-1].end
            if previous_end is None or start < previous_end:
                # 前一窗口向右开放（None）时任何后续起点都与之重叠；
                # 首尾相接（start == previous_end，半开语义）允许
                raise CustodyFeeQuoteError(
                    f"保管人 {schedule.custodian} 的费率窗口重叠："
                    f"{rates[-1].start.isoformat()} 起的窗口与 "
                    f"{start.isoformat()} 起的窗口相交"
                )
        rates.append(_Rate(start=start, end=end, hourly_rate=hourly_rate))
    return rates


def _exact_seconds(delta: timedelta) -> Fraction:
    """把时长精确换算为秒（含亚秒时间戳）：整数秒 + 微秒 / 1_000_000。"""
    return Fraction(delta.seconds + delta.days * 86400) + Fraction(delta.microseconds, 1_000_000)


def _fraction_amount(delta: timedelta, hourly_rate: Decimal) -> Fraction:
    """计算单段未舍入金额的精确有理数值：秒数 / 3600 × hourly_rate。"""
    return _exact_seconds(delta) / _SECONDS_PER_HOUR * Fraction(hourly_rate)


def _amount_decimal(value: Fraction) -> Decimal:
    """把精确有理金额转为高精度十进制（未舍入，原样保留可表示位数）。"""
    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PRECISION
        return ctx.divide(Decimal(value.numerator), Decimal(value.denominator))


def quote_custody_fee(request: CustodyFeeQuoteRequest) -> CustodyFeeQuoteResponse:
    """校验事件链并按保管人生效费率窗口试算保管费用。"""
    validation = validate_events(request.events)

    # 阶段 1：事件、settlement_until 与费率窗口不得混用时区
    timestamps = [request.settlement_until]
    timestamps.extend(event.timestamp for event in request.events)
    for schedule in request.rates:
        timestamps.extend(window.effective_from for window in schedule.windows)
        timestamps.extend(window.effective_to for window in schedule.windows)
    ensure_comparable(timestamps, CustodyFeeQuoteError)

    # 阶段 2：settlement_until 不得早于最后一个有效事件时间
    last_valid: Optional[datetime] = None
    for event, result in zip(request.events, validation.events):
        if result.status == "ok" and (last_valid is None or event.timestamp > last_valid):
            last_valid = event.timestamp
    if last_valid is not None and request.settlement_until < last_valid:
        raise CustodyFeeQuoteError(
            "settlement_until 早于最后一个有效事件时间："
            f"{request.settlement_until.isoformat()} 早于 {last_valid.isoformat()}"
        )

    # 阶段 3：校验并排序各保管人的费率窗口
    rates_by_custodian: Dict[str, List[_Rate]] = {}
    for schedule in request.rates:
        rates_by_custodian[schedule.custodian] = _validate_windows(schedule)

    # 阶段 4：生成保管区间，按保管人首次出现顺序分组
    intervals = _derive_custody_intervals(
        request.events, validation.events, request.settlement_until
    )
    order: List[str] = []
    by_custodian: Dict[str, List[_CustodyInterval]] = {}
    for interval in intervals:
        if interval.custodian not in by_custodian:
            by_custodian[interval.custodian] = []
            order.append(interval.custodian)
        by_custodian[interval.custodian].append(interval)

    # 阶段 5：按费率边界切分各保管区间，逐段计费或标记 unpriced
    priced: List[PricedCustodySegment] = []
    unpriced: List[UnpricedCustodySegment] = []
    subtotals: List[CustodianFeeQuote] = []
    for custodian in order:  # 保管人按首次出现顺序；同保管人区间按时间升序
        covers = [
            CoverInterval(start=rate.start, end=rate.end, payload=rate)
            for rate in rates_by_custodian.get(custodian, [])
        ]
        exact_total = Fraction(0)
        for interval in by_custodian[custodian]:
            for piece in split_by_cover(interval.start, interval.end, covers):
                delta = piece.end - piece.start
                seconds = delta.total_seconds()
                if piece.payload is None:
                    unpriced.append(
                        UnpricedCustodySegment(
                            custodian=custodian,
                            start=piece.start,
                            end=piece.end,
                            seconds=seconds,
                            custodian_event_index=interval.event_index,
                        )
                    )
                    continue
                rate: _Rate = piece.payload
                exact_amount = _fraction_amount(delta, rate.hourly_rate)
                exact_total += exact_amount
                priced.append(
                    PricedCustodySegment(
                        custodian=custodian,
                        start=piece.start,
                        end=piece.end,
                        seconds=seconds,
                        hourly_rate=rate.hourly_rate,
                        amount=_amount_decimal(exact_amount),
                        custodian_event_index=interval.event_index,
                    )
                )
        # 每位保管人先精确求和，再按 ROUND_HALF_UP 舍入到分
        subtotals.append(
            CustodianFeeQuote(
                custodian=custodian,
                subtotal=_amount_decimal(exact_total).quantize(_CENT, rounding=ROUND_HALF_UP),
            )
        )

    # total 由已舍入小计直接相加
    total = sum((item.subtotal for item in subtotals), Decimal("0.00"))

    return CustodyFeeQuoteResponse(
        settlement_until=request.settlement_until,
        validation=validation,
        segments=priced,
        unpriced=unpriced,
        custodians=subtotals,
        total=total,
    )
