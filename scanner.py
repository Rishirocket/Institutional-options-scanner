from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import *  # noqa: F403


Path(DATA_DIR).mkdir(exist_ok=True)  # noqa: F405
STATE_FILE = Path(DATA_DIR) / "state.json"  # noqa: F405

MIN_4H_RVOL = float(globals().get("MIN_4H_RVOL", 1.30))
ENTRY_BUFFER_ATR = float(globals().get("ENTRY_BUFFER_ATR", 0.05))
STOP_BUFFER_ATR = float(globals().get("STOP_BUFFER_ATR", 0.10))
MAX_UNDERLYING_RISK_PCT = float(globals().get("MAX_UNDERLYING_RISK_PCT", 3.0))
WATCH_DISTANCE_ATR = float(globals().get("WATCH_DISTANCE_ATR", 1.0))


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = df["Close"].shift()
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(length).mean()


def hv30(daily: pd.DataFrame) -> float:
    returns = np.log(daily["Close"] / daily["Close"].shift()).dropna()
    if len(returns) < 30:
        return np.nan
    return float(returns.tail(30).std() * math.sqrt(252))


def point_of_control(df: pd.DataFrame, bins: int = 30) -> float:
    """Volume-profile POC using typical price and traded volume."""
    sample = df.tail(120)
    prices = ((sample["High"] + sample["Low"] + sample["Close"]) / 3).dropna()
    volumes = sample.loc[prices.index, "Volume"].fillna(0)
    if len(prices) < 20 or float(volumes.sum()) <= 0:
        return np.nan
    histogram, edges = np.histogram(prices, bins=bins, weights=volumes)
    index = int(np.argmax(histogram))
    return float((edges[index] + edges[index + 1]) / 2)


def relative_strength_returns(
    stock: pd.DataFrame, spy: pd.DataFrame, periods: Tuple[int, ...] = (20, 63)
) -> Dict[int, float]:
    joined = pd.concat(
        [stock["Close"].rename("stock"), spy["Close"].rename("spy")], axis=1, join="inner"
    ).dropna()
    result = {}
    for period in periods:
        if len(joined) <= period:
            result[period] = np.nan
            continue
        stock_return = joined["stock"].iloc[-1] / joined["stock"].iloc[-period - 1] - 1
        spy_return = joined["spy"].iloc[-1] / joined["spy"].iloc[-period - 1] - 1
        result[period] = float(stock_return - spy_return)
    return result


def directional_volume_ratio(daily: pd.DataFrame, length: int = 20) -> float:
    sample = daily.tail(length).copy()
    previous = sample["Close"].shift()
    up_volume = float(sample.loc[sample["Close"] > previous, "Volume"].sum())
    down_volume = float(sample.loc[sample["Close"] < previous, "Volume"].sum())
    return up_volume / down_volume if down_volume > 0 else math.inf


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    wanted = ["Open", "High", "Low", "Close", "Volume"]
    return df[wanted].dropna(subset=["Open", "High", "Low", "Close"]).copy()


def completed_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="America/New_York")
    last_date = pd.Timestamp(df.index[-1]).date()
    if last_date == now.date() and now.time() < pd.Timestamp("16:00").time():
        return df.iloc[:-1].copy()
    return df.copy()


def weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        daily.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    now = pd.Timestamp.now(tz="America/New_York")
    if len(weekly) and now.weekday() < 4:
        weekly = weekly.iloc[:-1]
    elif len(weekly) and now.weekday() == 4 and now.time() < pd.Timestamp("16:00").time():
        weekly = weekly.iloc[:-1]
    return weekly


def session_four_hour(one_hour: pd.DataFrame) -> pd.DataFrame:
    """Build 9:30-13:30 and 13:30-16:00 ET session bars and drop an unfinished bar."""
    if one_hour.empty:
        return one_hour
    data = one_hour.copy()
    if data.index.tz is None:
        data.index = data.index.tz_localize("America/New_York")
    else:
        data.index = data.index.tz_convert("America/New_York")
    minute_of_day = data.index.hour * 60 + data.index.minute
    data = data[(minute_of_day >= 570) & (minute_of_day < 960)].copy()
    minute_of_day = data.index.hour * 60 + data.index.minute
    data["SessionDate"] = data.index.date
    data["Bucket"] = np.where(minute_of_day < 810, 0, 1)
    bars = (
        data.groupby(["SessionDate", "Bucket"])
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    timestamps = []
    for session_date, bucket in bars.index:
        start_time = "09:30" if bucket == 0 else "13:30"
        timestamps.append(pd.Timestamp(f"{session_date} {start_time}", tz="America/New_York"))
    bars.index = pd.DatetimeIndex(timestamps)
    bars["Bucket"] = [key[1] for key in bars.index.map(lambda x: (x.date(), 0 if x.hour < 13 else 1))]

    now = pd.Timestamp.now(tz="America/New_York")
    keep = []
    for timestamp in bars.index:
        if timestamp.date() < now.date():
            keep.append(True)
        elif timestamp.hour < 13:
            keep.append(now.time() >= pd.Timestamp("13:30").time())
        else:
            keep.append(now.time() >= pd.Timestamp("16:00").time())
    return bars.loc[keep].copy()


def pivot_values(df: pd.DataFrame, kind: str, wing: int = 2) -> List[Tuple[int, float]]:
    values = df["High"] if kind == "high" else df["Low"]
    points: List[Tuple[int, float]] = []
    for index in range(wing, len(df) - wing):
        window = values.iloc[index - wing : index + wing + 1]
        value = float(values.iloc[index])
        if kind == "high" and value == float(window.max()):
            points.append((index, value))
        elif kind == "low" and value == float(window.min()):
            points.append((index, value))
    return points


def structure_direction(df: pd.DataFrame) -> str:
    highs = pivot_values(df.tail(120), "high")
    lows = pivot_values(df.tail(120), "low")
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
            return "BULLISH"
        if highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
            return "BEARISH"
    return "MIXED"


def higher_timeframe_bias(df: pd.DataFrame) -> Dict[str, object]:
    close = df["Close"]
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    structure = structure_direction(df)
    bullish = close.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] and ema20.iloc[-1] > ema20.iloc[-4]
    bearish = close.iloc[-1] < ema20.iloc[-1] < ema50.iloc[-1] and ema20.iloc[-1] < ema20.iloc[-4]
    bias = "BULLISH" if bullish else "BEARISH" if bearish else "NEUTRAL"
    return {
        "bias": bias,
        "structure": structure,
        "close": float(close.iloc[-1]),
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
    }


def four_hour_bias(fourh: pd.DataFrame) -> Dict[str, object]:
    frame = fourh.copy()
    frame["ema9"] = ema(frame["Close"], 9)
    frame["ema21"] = ema(frame["Close"], 21)
    frame["ema50"] = ema(frame["Close"], 50)
    bullish = (
        frame["Close"].iloc[-1] > frame["ema9"].iloc[-1] > frame["ema21"].iloc[-1] > frame["ema50"].iloc[-1]
        and frame["ema21"].iloc[-1] > frame["ema21"].iloc[-4]
    )
    bearish = (
        frame["Close"].iloc[-1] < frame["ema9"].iloc[-1] < frame["ema21"].iloc[-1] < frame["ema50"].iloc[-1]
        and frame["ema21"].iloc[-1] < frame["ema21"].iloc[-4]
    )
    return {
        "bias": "BULLISH" if bullish else "BEARISH" if bearish else "NEUTRAL",
        "ema9": float(frame["ema9"].iloc[-1]),
        "ema21": float(frame["ema21"].iloc[-1]),
        "ema50": float(frame["ema50"].iloc[-1]),
    }


def same_bucket_rvol(fourh: pd.DataFrame) -> float:
    if len(fourh) < 12:
        return np.nan
    bucket = int(fourh["Bucket"].iloc[-1])
    history = fourh.iloc[:-1]
    comparable = history[history["Bucket"] == bucket].tail(20)
    average = float(comparable["Volume"].mean()) if len(comparable) >= 8 else np.nan
    return float(fourh["Volume"].iloc[-1] / average) if np.isfinite(average) and average > 0 else np.nan


def close_location(candle: pd.Series) -> float:
    spread = float(candle["High"] - candle["Low"])
    return float((candle["Close"] - candle["Low"]) / spread) if spread > 0 else 0.5


def entry_plan(fourh: pd.DataFrame, side: str) -> Optional[Dict[str, float | str]]:
    if len(fourh) < 60:
        return None
    sample = fourh.tail(120).copy()
    current_atr = float(atr(sample).iloc[-1])
    if not np.isfinite(current_atr) or current_atr <= 0:
        return None
    highs = pivot_values(sample.iloc[:-1], "high")
    lows = pivot_values(sample.iloc[:-1], "low")
    if not highs or not lows:
        return None
    last, previous = sample.iloc[-1], sample.iloc[-2]
    rvol = same_bucket_rvol(sample)

    if side == "CALL":
        structural_level = highs[-1][1]
        trigger = structural_level + ENTRY_BUFFER_ATR * current_atr
        structural_stop = lows[-1][1] - STOP_BUFFER_ATR * current_atr
        confirmed = (
            previous["Close"] <= trigger
            and last["Close"] > trigger
            and np.isfinite(rvol)
            and rvol >= MIN_4H_RVOL
            and close_location(last) >= 0.70
        )
        distance_atr = (trigger - float(last["Close"])) / current_atr
        status = "ENTRY" if confirmed else "WATCH" if 0 <= distance_atr <= WATCH_DISTANCE_ATR else "NONE"
        risk = trigger - structural_stop
        pt1, pt2 = trigger + risk, trigger + 2 * risk
    else:
        structural_level = lows[-1][1]
        trigger = structural_level - ENTRY_BUFFER_ATR * current_atr
        structural_stop = highs[-1][1] + STOP_BUFFER_ATR * current_atr
        confirmed = (
            previous["Close"] >= trigger
            and last["Close"] < trigger
            and np.isfinite(rvol)
            and rvol >= MIN_4H_RVOL
            and close_location(last) <= 0.30
        )
        distance_atr = (float(last["Close"]) - trigger) / current_atr
        status = "ENTRY" if confirmed else "WATCH" if 0 <= distance_atr <= WATCH_DISTANCE_ATR else "NONE"
        risk = structural_stop - trigger
        pt1, pt2 = trigger - risk, trigger - 2 * risk

    risk_pct = risk / trigger * 100
    if risk <= 0 or risk_pct > MAX_UNDERLYING_RISK_PCT or status == "NONE":
        return None
    return {
        "status": status,
        "level": structural_level,
        "trigger": trigger,
        "stop": structural_stop,
        "pt1": pt1,
        "pt2": pt2,
        "risk_pct": risk_pct,
        "rvol": rvol,
        "close_location": close_location(last),
    }


def pick_expiry(options: List[str], low: int, high: int) -> Optional[str]:
    today = date.today()
    candidates = []
    for expiry in options:
        try:
            dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
            if low <= dte <= high:
                candidates.append((abs(dte - (low + high) / 2), expiry))
        except Exception:
            continue
    return min(candidates)[1] if candidates else None


def black_scholes_delta(stock_price, strike, years, iv, call=True, rate=0.045):
    if stock_price <= 0 or strike <= 0 or years <= 0 or not np.isfinite(iv) or iv <= 0:
        return np.nan
    from math import erf, log, sqrt

    d1 = (log(stock_price / strike) + (rate + iv * iv / 2) * years) / (iv * sqrt(years))
    normal_d1 = 0.5 * (1 + erf(d1 / sqrt(2)))
    return normal_d1 if call else normal_d1 - 1


def option_rules(dte: int) -> Dict[str, float]:
    if dte <= 10:
        return {"delta": 0.50, "min_oi": 1000, "min_volume": 100, "max_spread": 8.0}
    if dte <= 18:
        return {"delta": 0.45, "min_oi": 750, "min_volume": 75, "max_spread": 10.0}
    return {"delta": 0.40, "min_oi": 500, "min_volume": 50, "max_spread": 12.0}


def select_option(
    chain: pd.DataFrame, stock_price: float, side: str, expiry: str
) -> Optional[pd.Series]:
    if chain.empty:
        return None
    dte = max((datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days, 1)
    rules = option_rules(dte)
    contracts = chain.copy()
    contracts["mid"] = (contracts["bid"].fillna(0) + contracts["ask"].fillna(0)) / 2
    contracts["spread_pct"] = (
        (contracts["ask"] - contracts["bid"]) / contracts["mid"].replace(0, np.nan) * 100
    )
    years = dte / 365
    contracts["calc_delta"] = contracts.apply(
        lambda row: black_scholes_delta(
            stock_price,
            float(row["strike"]),
            years,
            float(row.get("impliedVolatility", np.nan)),
            call=side == "CALL",
        ),
        axis=1,
    )
    contracts["abs_delta"] = contracts["calc_delta"].abs()
    contracts = contracts[
        (contracts["mid"] > 0.05)
        & (contracts["openInterest"].fillna(0) >= rules["min_oi"])
        & (contracts["volume"].fillna(0) >= rules["min_volume"])
        & (contracts["spread_pct"] <= rules["max_spread"])
        & contracts["abs_delta"].between(0.30, 0.70)
    ].copy()
    if contracts.empty:
        return None
    contracts["delta_distance"] = (contracts["abs_delta"] - rules["delta"]).abs()
    return contracts.sort_values(["delta_distance", "spread_pct"], ascending=True).iloc[0]


def earnings_before_expiry(ticker_object: yf.Ticker, expiry: str) -> bool:
    try:
        calendar = ticker_object.calendar
        if isinstance(calendar, pd.DataFrame) and "Earnings Date" in calendar.index:
            values = pd.to_datetime(calendar.loc["Earnings Date"].dropna()).date.tolist()
        elif isinstance(calendar, dict):
            raw = calendar.get("Earnings Date", [])
            parsed = pd.to_datetime(raw)
            values = [parsed.date()] if isinstance(parsed, pd.Timestamp) else parsed.date.tolist()
        else:
            return False
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        return any(date.today() <= value <= expiry_date for value in values)
    except Exception:
        return False


def next_earnings_date(ticker_object: yf.Ticker) -> Optional[date]:
    try:
        earnings = ticker_object.get_earnings_dates(limit=12)
        if earnings is not None and not earnings.empty:
            dates = pd.to_datetime(earnings.index)
            future = sorted({timestamp.date() for timestamp in dates if timestamp.date() >= date.today()})
            if future:
                return future[0]
    except Exception:
        pass
    try:
        calendar = ticker_object.get_calendar()
        raw = calendar.get("Earnings Date", []) if isinstance(calendar, dict) else []
        parsed = pd.to_datetime(raw)
        values = [parsed.date()] if isinstance(parsed, pd.Timestamp) else parsed.date.tolist()
        future = sorted({value for value in values if value >= date.today()})
        return future[0] if future else None
    except Exception:
        return None


def is_previous_business_day(event_date: date) -> bool:
    if event_date <= date.today():
        return False
    return int(np.busday_count(date.today().isoformat(), event_date.isoformat())) == 1


def grade_sentiment(grade: str) -> int:
    text = str(grade or "").lower()
    bullish = ("buy", "outperform", "overweight", "positive", "accumulate", "market perform +")
    bearish = ("sell", "underperform", "underweight", "negative", "reduce")
    if any(word in text for word in bullish):
        return 1
    if any(word in text for word in bearish):
        return -1
    return 0


def latest_analyst_actions(ticker_object: yf.Ticker) -> Tuple[List[str], int]:
    try:
        actions = ticker_object.get_upgrades_downgrades()
        if actions is None or actions.empty:
            return [], 0
        frame = actions.copy()
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=92)
        frame = frame[frame.index >= cutoff].sort_index(ascending=False)
        if frame.empty:
            return [], 0
        firm_column = "Firm" if "Firm" in frame.columns else "firm" if "firm" in frame.columns else None
        if firm_column:
            frame = frame.drop_duplicates(subset=[firm_column], keep="first")
        frame = frame.head(5)
        lines, score = [], 0
        for timestamp, row in frame.iterrows():
            firm = str(row.get("Firm", row.get("firm", "Unknown firm")))
            to_grade = str(row.get("ToGrade", row.get("toGrade", "Unrated")))
            action = str(row.get("Action", row.get("action", "reiterated")))
            score += grade_sentiment(to_grade)
            lines.append(f"{firm}: {action} → {to_grade} ({timestamp.date()})")
        return lines, score
    except Exception:
        return [], 0


def recommendation_summary(ticker_object: yf.Ticker) -> Tuple[str, int]:
    try:
        summary = ticker_object.get_recommendations_summary()
        if summary is None or summary.empty:
            return "Unavailable", 0
        row = summary.iloc[0]
        strong_buy = int(row.get("strongBuy", 0) or 0)
        buy = int(row.get("buy", 0) or 0)
        hold = int(row.get("hold", 0) or 0)
        sell = int(row.get("sell", 0) or 0)
        strong_sell = int(row.get("strongSell", 0) or 0)
        score = strong_buy * 2 + buy - sell - strong_sell * 2
        text = f"Strong Buy {strong_buy} | Buy {buy} | Hold {hold} | Sell {sell} | Strong Sell {strong_sell}"
        return text, score
    except Exception:
        return "Unavailable", 0


def eps_revision_score(ticker_object: yf.Ticker) -> Tuple[str, int]:
    try:
        revisions = ticker_object.get_eps_revisions()
        if revisions is None or revisions.empty:
            return "Unavailable", 0
        row = revisions.loc["0q"] if "0q" in revisions.index else revisions.iloc[0]
        up = int(row.get("upLast30days", 0) or 0)
        down = int(row.get("downLast30days", 0) or 0)
        return f"EPS revisions, last 30d: {up} up / {down} down", int(np.sign(up - down))
    except Exception:
        return "Unavailable", 0


def last_earnings_result(ticker_object: yf.Ticker) -> Tuple[str, int]:
    try:
        history = ticker_object.get_earnings_history()
        if history is None or history.empty:
            return "Unavailable", 0
        past = history[pd.to_datetime(history.index).date < date.today()]
        if past.empty:
            return "Unavailable", 0
        row = past.sort_index().iloc[-1]
        surprise = float(row.get("surprisePercent", np.nan))
        if not np.isfinite(surprise):
            return "Unavailable", 0
        return f"Last-quarter EPS surprise: {surprise:+.1f}%", int(np.sign(surprise))
    except Exception:
        return "Unavailable", 0


def implied_move(ticker_object: yf.Ticker, stock_price: float, earnings_date: date) -> Tuple[float, str]:
    try:
        expiries = sorted(
            expiry
            for expiry in ticker_object.options
            if datetime.strptime(expiry, "%Y-%m-%d").date() >= earnings_date
        )
        if not expiries:
            return np.nan, ""
        expiry = expiries[0]
        chain = ticker_object.option_chain(expiry)
        call = chain.calls.iloc[(chain.calls["strike"] - stock_price).abs().argsort()[:1]]
        put = chain.puts.iloc[(chain.puts["strike"] - stock_price).abs().argsort()[:1]]
        call_mid = float((call["bid"].iloc[0] + call["ask"].iloc[0]) / 2)
        put_mid = float((put["bid"].iloc[0] + put["ask"].iloc[0]) / 2)
        move = (call_mid + put_mid) / stock_price * 100
        return move, expiry
    except Exception:
        return np.nan, ""


def earnings_outlook_alert(
    ticker: str,
    ticker_object: yf.Ticker,
    event_date: date,
    stock_price: float,
    weekly_view: Dict,
    daily_view: Dict,
    fourh_view: Dict,
    spy_view: Dict,
    rs: Dict[int, float],
    poc: float,
    volume_ratio: float,
) -> str:
    analyst_lines, analyst_score = latest_analyst_actions(ticker_object)
    recommendation_text, recommendation_score = recommendation_summary(ticker_object)
    revision_text, revision_score = eps_revision_score(ticker_object)
    prior_text, prior_score = last_earnings_result(ticker_object)
    implied_pct, implied_expiry = implied_move(ticker_object, stock_price, event_date)

    technical_score = 0
    for view in (weekly_view, daily_view, fourh_view):
        technical_score += 1 if view["bias"] == "BULLISH" else -1 if view["bias"] == "BEARISH" else 0
    technical_score += 1 if rs[20] > 0 and rs[63] > 0 else -1 if rs[20] < 0 and rs[63] < 0 else 0
    technical_score += 1 if stock_price > poc else -1
    technical_score += 1 if volume_ratio > 1.10 else -1 if volume_ratio < 0.90 else 0
    total = technical_score + analyst_score + int(np.sign(recommendation_score)) + revision_score + prior_score
    outlook = "BULLISH" if total >= 4 else "BEARISH" if total <= -4 else "MIXED"
    analyst_outlook = "BULLISH" if analyst_score > 0 else "BEARISH" if analyst_score < 0 else "MIXED"
    expected_low = stock_price * (1 - implied_pct / 100) if np.isfinite(implied_pct) else np.nan
    expected_high = stock_price * (1 + implied_pct / 100) if np.isfinite(implied_pct) else np.nan
    analyst_text = "\n".join(f"• {line}" for line in analyst_lines) if analyst_lines else "• No recent actions available"
    implied_text = (
        f"Options-implied move: ±{implied_pct:.1f}% → approximately ${expected_low:.2f}–${expected_high:.2f} (using {implied_expiry})"
        if np.isfinite(implied_pct)
        else "Options-implied move: unavailable"
    )
    return "\n".join(
        [
            f"📅 EARNINGS OUTLOOK | {ticker} | {event_date}",
            f"Evidence balance: {outlook} | Score {total:+d} (not a price prediction)",
            f"Weekly {weekly_view['bias']} | Daily {daily_view['bias']} | 4H {fourh_view['bias']} | SPY Daily {spy_view['bias']}",
            f"RS vs SPY: 20d {rs[20]*100:+.2f}% | 63d {rs[63]*100:+.2f}%",
            f"Price ${stock_price:.2f} vs 4H POC ${poc:.2f} | Up/down volume ratio {volume_ratio:.2f}",
            implied_text,
            revision_text,
            prior_text,
            f"Recommendation summary: {recommendation_text}",
            f"Latest-five analyst-action balance: {analyst_outlook} (not accuracy-ranked)",
            "Latest analyst actions from the last quarter:",
            analyst_text,
            "Earnings can gap beyond the implied range. Do not treat this outlook as a guaranteed CALL or PUT direction.",
        ]
    )


def chain_iv_percentile(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    iv = pd.concat([calls, puts])["impliedVolatility"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(iv) < 10:
        return np.nan
    current, low, high = float(iv.median()), float(iv.quantile(0.05)), float(iv.quantile(0.95))
    if high <= low:
        return np.nan
    return max(0.0, min(100.0, 100 * (current - low) / (high - low)))


def build_alert(
    ticker: str,
    side: str,
    dte_label: str,
    expiry: str,
    stock_price: float,
    option: pd.Series,
    weekly: Dict,
    daily: Dict,
    fourh: Dict,
    plan: Dict,
    iv_percentile: float,
    historical_volatility: float,
    spy_view: Dict,
    rs: Dict[int, float],
    poc: float,
    volume_ratio: float,
) -> str:
    dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
    strike = float(option["strike"])
    mid = float(option["mid"])
    premium_risk_cap = 0.30 if dte <= 10 else 0.25 if dte <= 18 else 0.20
    premium_stop = mid * (1 - premium_risk_cap)
    icon = "🟢" if side == "CALL" else "🔴"
    action = "ENTER" if plan["status"] == "ENTRY" else "WATCH"
    direction_word = "above" if side == "CALL" else "below"
    return "\n".join(
        [
            f"{icon} {action} {side} | {ticker} | {dte_label}",
            f"Weekly: {weekly['bias']} ({weekly['structure']}) | Daily: {daily['bias']} ({daily['structure']}) | 4H: {fourh['bias']}",
            f"SPY Daily: {spy_view['bias']} | RS vs SPY: 20d {rs[20]*100:+.2f}% / 63d {rs[63]*100:+.2f}%",
            f"Underlying: ${stock_price:.2f} | Structural level: ${plan['level']:.2f}",
            f"4H POC: ${poc:.2f} | Price is {'above' if stock_price > poc else 'below'} POC | Up/down volume ratio: {volume_ratio:.2f}",
            f"Entry: {direction_word} ${plan['trigger']:.2f} on a completed 4H close",
            f"4H volume: {plan['rvol']:.2f}x same-session average | Close strength: {plan['close_location']*100:.0f}%",
            f"Underlying stop: ${plan['stop']:.2f} | PT1: ${plan['pt1']:.2f} (1R) | PT2: ${plan['pt2']:.2f} (2R)",
            f"Contract: {ticker} ${strike:.0f} {side} | Exp: {expiry} | DTE: {dte}",
            f"Option mid: ${mid:.2f} | Delta: {float(option['calc_delta']):.2f} | Spread: {float(option['spread_pct']):.1f}%",
            f"Option volume/OI: {float(option.get('volume', 0) or 0):.0f}/{float(option.get('openInterest', 0) or 0):.0f}",
            f"Chain IV percentile: {iv_percentile:.1f}% | HV30: {historical_volatility*100:.1f}%",
            f"Premium risk cap: approximately ${premium_stop:.2f} (-{premium_risk_cap*100:.0f}%); underlying stop remains primary.",
            "No entry before the completed 4H confirmation. Check live quote and fill before trading.",
        ]
    )


def analyze_ticker(ticker: str, spy_daily: pd.DataFrame) -> List[Tuple[str, str]]:
    signals: List[Tuple[str, str]] = []
    ticker_object = yf.Ticker(ticker)
    daily = completed_daily(clean_history(ticker_object.history(period="3y", interval="1d", auto_adjust=True)))
    one_hour = clean_history(ticker_object.history(period="180d", interval="1h", auto_adjust=True))
    fourh = session_four_hour(one_hour)
    weekly = weekly_from_daily(daily)
    if len(daily) < 150 or len(weekly) < 55 or len(fourh) < 60:
        print(f"{ticker}: skipped - insufficient weekly/daily/4H history")
        return signals

    stock_price = float(fourh["Close"].iloc[-1])
    average_dollar_volume = float((daily["Close"] * daily["Volume"]).tail(30).mean())
    if average_dollar_volume < 20_000_000 or stock_price < 10:
        print(f"{ticker}: skipped - liquidity/price")
        return signals

    weekly_view = higher_timeframe_bias(weekly)
    daily_view = higher_timeframe_bias(daily)
    fourh_view = four_hour_bias(fourh)
    spy_view = higher_timeframe_bias(spy_daily)
    rs = relative_strength_returns(daily, spy_daily)
    poc = point_of_control(fourh)
    volume_ratio = directional_volume_ratio(daily)
    if not all(np.isfinite(value) for value in (rs[20], rs[63], poc, volume_ratio)):
        print(f"{ticker}: skipped - SPY/RS/POC/volume data unavailable")
        return signals

    event_date = next_earnings_date(ticker_object)
    if event_date and is_previous_business_day(event_date):
        earnings_alert = earnings_outlook_alert(
            ticker,
            ticker_object,
            event_date,
            stock_price,
            weekly_view,
            daily_view,
            fourh_view,
            spy_view,
            rs,
            poc,
            volume_ratio,
        )
        earnings_key = f"{date.today()}:{ticker}:EARNINGS_OUTLOOK:{event_date}"
        signals.append((earnings_key, earnings_alert))

    if weekly_view["bias"] != daily_view["bias"] or weekly_view["bias"] == "NEUTRAL":
        print(f"{ticker}: no trade - weekly/daily conflict ({weekly_view['bias']}/{daily_view['bias']})")
        return signals
    required_bias = weekly_view["bias"]
    if weekly_view["structure"] != required_bias or daily_view["structure"] != required_bias:
        print(
            f"{ticker}: no trade - price structure not confirmed "
            f"(weekly {weekly_view['structure']} / daily {daily_view['structure']})"
        )
        return signals

    if required_bias == "BULLISH":
        market_ok = spy_view["bias"] != "BEARISH" or rs[63] >= 0.03
        rs_ok = rs[20] > 0 and rs[63] > 0
        poc_ok = stock_price > poc
        volume_ok = volume_ratio >= 1.10
    else:
        market_ok = spy_view["bias"] != "BULLISH" or rs[63] <= -0.03
        rs_ok = rs[20] < 0 and rs[63] < 0
        poc_ok = stock_price < poc
        volume_ok = volume_ratio <= (1 / 1.10)
    if not market_ok:
        print(f"{ticker}: no trade - SPY regime opposes {required_bias} setup")
        return signals
    if not rs_ok:
        print(f"{ticker}: no trade - relative strength does not confirm {required_bias}")
        return signals
    if not poc_ok:
        print(f"{ticker}: no trade - price is on the wrong side of 4H POC ${poc:.2f}")
        return signals
    if not volume_ok:
        print(f"{ticker}: no trade - directional volume ratio {volume_ratio:.2f} does not confirm")
        return signals
    if fourh_view["bias"] != required_bias:
        print(f"{ticker}: no trade - 4H not aligned ({required_bias}/{fourh_view['bias']})")
        return signals
    side = "CALL" if required_bias == "BULLISH" else "PUT"
    plan = entry_plan(fourh, side)
    if not plan:
        print(f"{ticker}: aligned but not near a valid 4H structural entry")
        return signals

    options = list(ticker_object.options or [])
    historical_volatility = hv30(daily)
    for dte_label, (low, high) in DTE_BUCKETS.items():  # noqa: F405
        expiry = pick_expiry(options, low, high)
        if not expiry:
            continue
        if earnings_before_expiry(ticker_object, expiry):
            print(f"{ticker} {dte_label}: skipped - earnings before expiration")
            continue
        try:
            chain = ticker_object.option_chain(expiry)
            calls, puts = chain.calls.copy(), chain.puts.copy()
        except Exception as error:
            print(f"{ticker}: option chain failed {expiry}: {error}")
            continue
        chain_frame = calls if side == "CALL" else puts
        option = select_option(chain_frame, stock_price, side, expiry)
        if option is None:
            print(f"{ticker} {dte_label}: no liquid contract near target delta")
            continue
        iv_percentile = chain_iv_percentile(calls, puts)
        if not np.isfinite(iv_percentile) or not np.isfinite(historical_volatility):
            continue
        alert = build_alert(
            ticker,
            side,
            dte_label,
            expiry,
            stock_price,
            option,
            weekly_view,
            daily_view,
            fourh_view,
            plan,
            iv_percentile,
            historical_volatility,
            spy_view,
            rs,
            poc,
            volume_ratio,
        )
        key = f"{date.today()}:{ticker}:{side}:{dte_label}:{plan['status']}:{plan['trigger']:.2f}"
        signals.append((key, alert))
    return signals


def send_alert(text: str) -> bool:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print(text)
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:3900], "disable_web_page_preview": True},
            timeout=15,
        )
        print(f"Telegram status: {response.status_code} {response.text[:200]}")
        return response.ok
    except Exception as error:
        print(f"Telegram failed: {error}\n{text}")
        return False


def load_previous_state() -> Dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def run_scan(send_test: bool = False) -> Dict:
    batch = sorted({ticker for ticker in WHITELIST if ticker not in BLACKLIST})[:MAX_TICKERS_PER_RUN]  # noqa: F405
    print(f"Running {len(batch)} tickers:")
    print(batch)
    if send_test:
        send_alert("✅ Telegram test from Weekly/Daily/4H Options Bot")

    spy_daily = completed_daily(
        clean_history(yf.Ticker("SPY").history(period="3y", interval="1d", auto_adjust=True))
    )
    if len(spy_daily) < 150:
        raise RuntimeError("SPY history unavailable; scan stopped because market/RS filters cannot be calculated")

    previous = load_previous_state()
    sent_keys = dict(previous.get("sent_signal_keys", {}))
    today_text = date.today().isoformat()
    sent_keys = {key: value for key, value in sent_keys.items() if str(value) >= today_text}
    alert_texts: List[str] = []
    errors: List[str] = []
    suppressed = 0
    for ticker in batch:
        try:
            print(f"Checking {ticker}...")
            for key, alert in analyze_ticker(ticker, spy_daily):
                if key in sent_keys:
                    suppressed += 1
                    continue
                send_alert(alert)
                sent_keys[key] = today_text
                alert_texts.append(alert)
        except Exception as error:
            message = f"{ticker}: {type(error).__name__}: {error}"
            print(message)
            errors.append(message)
        time.sleep(SLEEP_BETWEEN_TICKERS)  # noqa: F405

    result = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "strategy": "Volume + Weekly/Daily/4H trend + EMA + SPY/RS + 4H POC",
        "tickers": len(batch),
        "alerts": len(alert_texts),
        "duplicates_suppressed": suppressed,
        "errors": errors[-20:],
        "alert_texts": alert_texts[-20:],
        "sent_signal_keys": sent_keys,
    }
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(STATE_FILE)
    print(f"Done. Total alerts: {len(alert_texts)} | Suppressed: {suppressed}")
    return result


if __name__ == "__main__":
    run_scan(send_test=os.getenv("SEND_TEST", "0") == "1")