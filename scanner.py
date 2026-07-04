from __future__ import annotations

import json, math, os, time
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import *

Path(DATA_DIR).mkdir(exist_ok=True)
STATE_FILE = Path(DATA_DIR) / "state.json"


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def hv30(daily: pd.DataFrame) -> float:
    r = np.log(daily["Close"] / daily["Close"].shift(1)).dropna()
    if len(r) < 30:
        return np.nan
    return float(r.tail(30).std() * math.sqrt(252))


def volume_profile_shelves(df: pd.DataFrame, bins: int = 24, top_n: int = 3) -> List[float]:
    if df.empty:
        return []
    prices = ((df["High"] + df["Low"] + df["Close"]) / 3).dropna()
    vols = df.loc[prices.index, "Volume"].fillna(0)
    if len(prices) < 5 or vols.sum() <= 0:
        return []
    hist, edges = np.histogram(prices, bins=bins, weights=vols)
    idx = np.argsort(hist)[-top_n:][::-1]
    return [float((edges[i] + edges[i+1]) / 2) for i in idx]


def pick_expiry(options: List[str], lo: int, hi: int) -> Optional[str]:
    today = date.today()
    candidates = []
    for e in options:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            if lo <= dte <= hi:
                candidates.append((abs(dte - ((lo + hi) // 2)), e, dte))
        except Exception:
            continue
    return sorted(candidates)[0][1] if candidates else None


def black_scholes_delta(S, K, T, iv, call=True, r=0.045):
    if S <= 0 or K <= 0 or T <= 0 or not np.isfinite(iv) or iv <= 0:
        return np.nan
    from math import log, sqrt, erf
    d1 = (log(S/K) + (r + iv*iv/2) * T) / (iv * sqrt(T))
    nd1 = 0.5 * (1 + erf(d1 / sqrt(2)))
    return nd1 if call else nd1 - 1


def iv_rank_from_chain(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    iv = pd.concat([calls, puts])["impliedVolatility"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(iv) < 10:
        return np.nan
    cur, lo, hi = float(iv.median()), float(iv.quantile(.05)), float(iv.quantile(.95))
    if hi <= lo:
        return np.nan
    return max(0, min(100, 100 * (cur - lo) / (hi - lo)))


def select_option(chain: pd.DataFrame, S: float, side: str, dte_label: str) -> Optional[pd.Series]:
    x = chain.copy()
    if x.empty:
        return None
    x["mid"] = (x["bid"].fillna(0) + x["ask"].fillna(0)) / 2
    x["spread_pct"] = ((x["ask"] - x["bid"]) / x["mid"].replace(0, np.nan)) * 100
    x = x[(x["mid"] > 0.05) & (x["openInterest"].fillna(0) >= 500) & (x["volume"].fillna(0) >= 50)]
    if x.empty:
        return None
    if dte_label == "7DTE":
        target_delta = 0.50
    elif dte_label == "30DTE":
        target_delta = 0.40
    else:
        target_delta = 0.65
    x["dist"] = (x["strike"] - S).abs()
    return x.sort_values(["dist", "spread_pct"]).iloc[0]


def score_setup(side: str, dte_label: str, S: float, daily: pd.DataFrame, fourh: pd.DataFrame, monthly: pd.DataFrame, ivr: float, hv: float, med_iv: float, opt: pd.Series) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []

    monthly_close = monthly["Close"]
    m_ema50 = float(ema(monthly_close, 50).iloc[-1])
    m_ema200 = float(ema(monthly_close, 200).iloc[-1]) if len(monthly) >= 200 else np.nan
    m_green = bool(monthly["Close"].iloc[-1] >= monthly["Open"].iloc[-1])
    m_shelves = volume_profile_shelves(monthly, top_n=3)

    f = fourh.copy()
    f["ema9"] = ema(f.Close, 9)
    f["ema21"] = ema(f.Close, 21)
    f["ema50"] = ema(f.Close, 50)
    f["rsi"] = rsi(f.Close, 14)
    f_shelves = volume_profile_shelves(f.tail(100), top_n=3)

    d = daily.copy()
    d["ema20"] = ema(d.Close, 20)
    d["ema50"] = ema(d.Close, 50)
    d["bb_mid"] = d.Close.rolling(20).mean()
    d["bb_std"] = d.Close.rolling(20).std()
    d["bb_hi"] = d.bb_mid + 2 * d.bb_std
    d["bb_lo"] = d.bb_mid - 2 * d.bb_std
    d_atr = float(atr(d).iloc[-1])

    if side == "CALL":
        if S > m_ema50 and m_green:
            score += 20; reasons.append("Monthly bias bullish: price above EMA50 and green candle")
        if S > d.ema20.iloc[-1] > d.ema50.iloc[-1]:
            score += 20; reasons.append("Daily trend bullish: close > EMA20 > EMA50")
        if f.ema9.iloc[-1] > f.ema21.iloc[-1] > f.ema50.iloc[-1]:
            score += 20; reasons.append("4H EMA stack bullish")
        if f_shelves and S > min(f_shelves, key=lambda x: abs(x-S)):
            score += 10; reasons.append("Price holding above nearest 4H VP shelf")
        if 20 <= ivr <= 55 and np.isfinite(hv) and med_iv <= hv * 1.30:
            score += 15; reasons.append("IV acceptable versus HV30")
        if float(opt.get("volume", 0) or 0) > float(opt.get("openInterest", 0) or 0) * 0.5:
            score += 15; reasons.append("Option volume active versus OI")
    else:
        if S < m_ema50 or not m_green:
            score += 20; reasons.append("Monthly bias weak: below EMA50 or red candle")
        if S < d.ema20.iloc[-1] < d.ema50.iloc[-1]:
            score += 20; reasons.append("Daily trend bearish: close < EMA20 < EMA50")
        if f.ema9.iloc[-1] < f.ema21.iloc[-1] < f.ema50.iloc[-1]:
            score += 20; reasons.append("4H EMA stack bearish")
        if f_shelves and S < min(f_shelves, key=lambda x: abs(x-S)):
            score += 10; reasons.append("Price below nearest 4H VP shelf")
        if 20 <= ivr <= 60 and np.isfinite(hv) and med_iv <= hv * 1.50:
            score += 15; reasons.append("IV acceptable for bearish premium")
        if float(opt.get("volume", 0) or 0) > float(opt.get("openInterest", 0) or 0) * 0.5:
            score += 15; reasons.append("Option volume active versus OI")

    return min(score, 100), reasons


def build_alert(ticker: str, side: str, dte_label: str, expiry: str, S: float, opt: pd.Series, score: int, reasons: List[str], daily: pd.DataFrame) -> str:
    dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
    d_atr = float(atr(daily).iloc[-1])
    strike = float(opt["strike"])
    mid = float(opt.get("mid", (opt.get("bid", 0) + opt.get("ask", 0)) / 2) or 0)

    if side == "CALL":
        direction = "🟢 CALL OPTION SETUP"
        stock_stop = S - 1.25 * d_atr if dte_label == "7DTE" else S - 1.5 * d_atr
        target_1 = S + 1.5 * d_atr
        target_2 = S + 3.0 * d_atr
    else:
        direction = "🔴 PUT OPTION SETUP"
        stock_stop = S + 1.25 * d_atr if dte_label == "7DTE" else S + 1.5 * d_atr
        target_1 = S - 1.5 * d_atr
        target_2 = S - 3.0 * d_atr

    premium_stop = mid * (0.60 if dte_label == "7DTE" else 0.50)
    premium_target = mid * 1.75

    return "\n".join([
        f"🚨 {direction} | {ticker} | {dte_label}",
        f"Score: {score}/100 | Exp: {expiry} | DTE: {dte}",
        f"Stock: ${S:.2f} | Strike: {strike:.0f} | Est option mid: ${mid:.2f}",
        f"Entry rule: enter only if price confirms above/below current trigger level.",
        f"Stock stop: ${stock_stop:.2f} | T1: ${target_1:.2f} | T2: ${target_2:.2f}",
        f"Option stop: ${premium_stop:.2f} | Option target: ${premium_target:.2f}",
        "Why: " + "; ".join(reasons[:5]),
        "Risk: setup alert only, verify fill, news, earnings, spread, and chain before trading.",
    ])


def analyze_ticker(ticker: str) -> List[str]:
    alerts: List[str] = []
    tk = yf.Ticker(ticker)
    daily = tk.history(period="1y", interval="1d", auto_adjust=False)
    monthly = tk.history(period="10y", interval="1mo", auto_adjust=False)
    oneh = tk.history(period="90d", interval="1h", auto_adjust=False)
    if daily.empty or monthly.empty or oneh.empty or len(daily) < 80 or len(monthly) < 60:
        print(f"{ticker}: skipped - not enough history")
        return alerts

    fourh = oneh.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    S = float(daily["Close"].iloc[-1])
    adv = float(daily["Volume"].tail(30).mean())
    if adv < 1_000_000 or S < 10:
        print(f"{ticker}: skipped - liquidity/price")
        return alerts

    options = list(tk.options or [])
    if not options:
        print(f"{ticker}: skipped - no options")
        return alerts

    hv = hv30(daily)
    for dte_label, (lo, hi) in DTE_BUCKETS.items():
        expiry = pick_expiry(options, lo, hi)
        if not expiry:
            continue
        try:
            chain = tk.option_chain(expiry)
            calls, puts = chain.calls.copy(), chain.puts.copy()
        except Exception as e:
            print(f"{ticker}: option chain failed {expiry}: {e}")
            continue
        if calls.empty or puts.empty:
            continue
        ivr = iv_rank_from_chain(calls, puts)
        if not np.isfinite(ivr):
            continue

        for side, chain_df in [("CALL", calls), ("PUT", puts)]:
            opt = select_option(chain_df, S, side, dte_label)
            if opt is None:
                continue
            med_iv = float(opt.get("impliedVolatility", np.nan))
            if not np.isfinite(med_iv):
                continue
            score, reasons = score_setup(side, dte_label, S, daily, fourh, monthly, ivr, hv, med_iv, opt)
            if score >= ALERT_SCORE_MIN:
                alerts.append(build_alert(ticker, side, dte_label, expiry, S, opt, score, reasons, daily))
    return alerts


def send_alert(text: str) -> bool:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print(text)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:3900], "disable_web_page_preview": True},
            timeout=15,
        )
        print(f"Telegram status: {r.status_code} {r.text[:200]}")
        return r.ok
    except Exception as e:
        print(f"Telegram failed: {e}\n{text}")
        return False


def run_scan(send_test: bool = False) -> Dict:
    batch = sorted(set([t for t in WHITELIST if t not in BLACKLIST]))[:MAX_TICKERS_PER_RUN]
    print(f"Running {len(batch)} tickers:")
    print(batch)
    if send_test:
        send_alert("👋 Hi from Institutional Options Scanner")
    all_alerts: List[str] = []
    errors: List[str] = []
    for ticker in batch:
        try:
            print(f"Checking {ticker}...")
            alerts = analyze_ticker(ticker)
            for a in alerts:
                send_alert(a)
            all_alerts.extend(alerts)
        except Exception as e:
            msg = f"{ticker}: {type(e).__name__}: {e}"
            print(msg)
            errors.append(msg)
        time.sleep(SLEEP_BETWEEN_TICKERS)
    result = {"time": datetime.now().isoformat(timespec="seconds"), "tickers": len(batch), "alerts": len(all_alerts), "errors": errors[-20:], "alert_texts": all_alerts[-10:]}
    STATE_FILE.write_text(json.dumps(result, indent=2))
    print(f"Done. Total alerts: {len(all_alerts)}")
    return result


if __name__ == "__main__":
    run_scan(send_test=os.getenv("SEND_TEST", "0") == "1")
