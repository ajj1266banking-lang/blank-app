<#
create_and_b64_zip.ps1

Creates a trading_app project under a parent directory (default: $env:USERPROFILE\Documents\GitHub),
zips the project to trading_app.zip and writes a base64-encoded version trading_app.zip.b64.

Usage (PowerShell):
  # From an elevated or normal PowerShell prompt:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\create_and_b64_zip.ps1
  # Or to specify a different parent directory:
  .\create_and_b64_zip.ps1 -RootParent "C:\Users\aj\Documents\GitHub" -ProjectName "trading_app"

#>

param(
    [string]$RootParent = "$env:USERPROFILE\Documents\GitHub",
    [string]$ProjectName = "trading_app"
)

$Root = Join-Path -Path $RootParent -ChildPath $ProjectName
$ZipPath = Join-Path -Path $RootParent -ChildPath ("$ProjectName.zip")
$B64Path = "$ZipPath.b64"

Write-Host "Creating project at: $Root"
if (Test-Path $Root) {
    Write-Host "Removing existing folder $Root"
    Remove-Item -Recurse -Force $Root
}

# create directories
New-Item -ItemType Directory -Path $Root -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Root "tests") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Root "data") -Force | Out-Null

function Write-ProjectFile {
    param(
        [string]$RelativePath,
        [string]$Content
    )
    $FullPath = Join-Path -Path $Root -ChildPath $RelativePath
    $Dir = Split-Path -Path $FullPath -Parent
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    }
    $Content | Out-File -FilePath $FullPath -Encoding UTF8
    Write-Host "Wrote $RelativePath"
}

# Write files (here-strings preserve contents)

Write-ProjectFile "app.py" @'
import os
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np

import storage
import analytics
import charts
import montecarlo

st.set_page_config(page_title="Personal Trading System", layout="wide")

st.title("Personal Trading System")

# Simple "authentication": username in sidebar
st.sidebar.header("User")
username = st.sidebar.text_input("Username", value="default").strip()
if username == "":
    st.sidebar.error("Enter a username to continue")
    st.stop()

# Data controls
st.sidebar.header("Storage")
st.sidebar.markdown("Data stored locally in SQLite (`data/trades.db`).")

# Add trade form
st.sidebar.header("Add a Trade")
with st.sidebar.form("trade_form", clear_on_submit=True):
    t_date = st.date_input("Date", value=date.today())
    t_symbol = st.text_input("Symbol")
    t_strategy = st.text_input("Strategy", value="Unspecified")
    t_pnl = st.number_input("P&L", value=0.0, format="%.2f")
    submitted = st.form_submit_button("Save trade")
    if submitted:
        row = {
            "date": pd.to_datetime(t_date).strftime("%Y-%m-%d"),
            "symbol": t_symbol.strip(),
            "strategy": t_strategy.strip() or "Unspecified",
            "pnl": float(t_pnl),
        }
        storage.save_trade(row, user=username)
        st.success("Trade saved")

# Main area: tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Strategy Stats", "Advanced Analytics"])

with tab1:
    st.subheader(f"Equity curve — user: {username}")
    curve = charts.equity_curve(user=username)
    if curve is None or curve.empty:
        st.info("No trades yet — add trades using the sidebar form.")
    else:
        st.line_chart(curve)

    st.subheader("Trades")
    df = storage.load_trades(user=username)
    if df.empty:
        st.info("No trades recorded yet for this user.")
    else:
        st.dataframe(df.sort_values("date", ascending=False))
        csv = storage.export_csv(user=username)
        st.download_button("Download trades CSV", data=csv, file_name=f"{username}_trades.csv", mime="text/csv")

with tab2:
    st.subheader("Strategy statistics")
    stats = analytics.strategy_stats(user=username)
    if stats.empty:
        st.info("No trades to compute stats from.")
    else:
        st.dataframe(stats)

with tab3:
    st.subheader("Advanced analytics")
    df = storage.load_trades(user=username)
    if df.empty:
        st.info("No trades to compute analytics.")
    else:
        # Allow customizing initial capital for return-based metrics
        initial_capital = st.number_input("Initial capital (for return-based metrics)", min_value=1.0, value=100000.0, step=1000.0, format="%.2f")

        win_rate = analytics.win_rate(user=username)
        max_dd = analytics.max_drawdown(user=username)
        sharpe = analytics.sharpe_ratio(user=username, initial_capital=initial_capital)

        st.metric("Win rate", f"{win_rate:.2%}" if win_rate is not None else "—")
        st.metric("Max drawdown (historic)", f"{max_dd:.2%}" if max_dd is not None else "—")
        st.metric("Sharpe-like (annualized)", f"{sharpe:.2f}" if sharpe is not None else "—")

        # New core analytics
        cagr_val = analytics.cagr(user=username, initial_capital=initial_capital)
        ann_vol = analytics.annual_volatility(user=username, initial_capital=initial_capital)
        sortino = analytics.sortino_ratio(user=username, initial_capital=initial_capital)
        calmar = analytics.calmar_ratio(user=username, initial_capital=initial_capital)
        expectancy = analytics.expectancy_per_trade(user=username)
        streak_info = analytics.streaks(user=username)
        dd_duration = analytics.drawdown_duration(user=username)

        # Rolling Sharpe window selector: allow 3-10 days (default 3)
        rs_window = st.slider("Rolling Sharpe window (days)", min_value=3, max_value=10, value=3)
        rolling_sharpe = analytics.rolling_sharpe(user=username, window=rs_window, initial_capital=initial_capital)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", f"{cagr_val:.2%}" if cagr_val is not None else "—")
        c2.metric("Annual vol", f"{ann_vol:.2%}" if ann_vol is not None else "—")
        c3.metric("Sortino", f"{sortino:.2f}" if sortino is not None else "—")
        c4.metric("Calmar", f"{calmar:.2f}" if calmar is not None else "—")

        c5, c6, c7 = st.columns(3)
        c5.metric("Expectancy / trade", f"{expectancy:.2f}" if expectancy is not None else "—")
        c6.metric("Max drawdown duration (days)", f"{dd_duration}" if dd_duration is not None else "—")
        c7.metric("Current win streak", f"{streak_info.get('current_win_streak', 0)}")

        st.markdown("Streaks and additional stats")
        st.write(streak_info)

        st.markdown(f"Rolling Sharpe ({rs_window}-day window)")
        if rolling_sharpe is None or rolling_sharpe.empty:
            st.info("Not enough data for rolling Sharpe with the chosen window.")
        else:
            st.line_chart(rolling_sharpe)

        st.markdown("Equity curve with drawdown (cumulative P&L and running drawdown).")
        eq = charts.equity_curve(user=username)
        st.line_chart(eq)

        st.markdown("Per-trade P&L distribution")
        st.bar_chart(df["pnl"])
'@

Write-ProjectFile "storage.py" @'
import os
import sqlite3
from typing import Optional

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_FILE = os.path.join(DATA_DIR, "trades.db")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def _ensure_db(db_file: Optional[str] = None):
    db = db_file or DB_FILE
    conn = sqlite3.connect(db)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                date TEXT,
                symbol TEXT,
                strategy TEXT,
                pnl REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# Initialize DB on import
_ensure_db()


def set_db_file(path: str):
    global DB_FILE
    DB_FILE = path
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    _ensure_db(DB_FILE)


def _get_conn():
    return sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)


def load_trades(user: Optional[str] = None) -> pd.DataFrame:
    conn = _get_conn()
    try:
        if user:
            df = pd.read_sql_query("SELECT * FROM trades WHERE user = ? ORDER BY date DESC", conn, params=(user,))
        else:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY date DESC", conn)
    finally:
        conn.close()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "pnl" in df.columns:
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    return df


def save_trade(row: dict, user: str):
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO trades (user, date, symbol, strategy, pnl) VALUES (?, ?, ?, ?, ?)",
            (
                user,
                row.get("date"),
                row.get("symbol"),
                row.get("strategy"),
                float(row.get("pnl", 0.0)),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def export_csv(user: Optional[str] = None) -> bytes:
    df = load_trades(user=user)
    return df.to_csv(index=False).encode("utf-8")
'@

Write-ProjectFile "charts.py" @'
from storage import load_trades
import pandas as pd


def equity_curve(user: str = None) -> pd.Series | None:
    df = load_trades(user=user)
    if df.empty or "date" not in df.columns:
        return None
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return None
    df = df.sort_values("date")
    daily = df.groupby("date", as_index=True)["pnl"].sum().sort_index()
    cum = daily.cumsum()
    if not isinstance(cum.index, pd.DatetimeIndex):
        cum.index = pd.to_datetime(cum.index)
    return cum
'@

Write-ProjectFile "analytics.py" @'
import math
from typing import Optional, Dict

import numpy as np
import pandas as pd

from storage import load_trades


def strategy_stats(user: str = None) -> pd.DataFrame:
    df = load_trades(user=user)
    if df.empty or "strategy" not in df.columns:
        return pd.DataFrame(columns=["strategy", "trades", "total_pnl", "avg_pnl"])

    grouped = df.groupby("strategy", dropna=False)
    stats = grouped.agg(
        trades=("strategy", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).reset_index()
    stats["strategy"] = stats["strategy"].fillna("Unspecified")
    return stats


def win_rate(user: str = None) -> Optional[float]:
    df = load_trades(user=user)
    if df.empty:
        return None
    wins = (df["pnl"] > 0).sum()
    return wins / len(df)


def max_drawdown(user: str = None) -> Optional[float]:
    df = load_trades(user=user)
    if df.empty or "date" not in df.columns:
        return None
    daily = df.groupby("date", as_index=True)["pnl"].sum().sort_index()
    if daily.empty:
        return None
    cum = daily.cumsum()
    running_max = cum.cummax()
    drawdown = (running_max - cum) / running_max.replace(0, np.nan)
    max_dd = drawdown.max()
    if pd.isna(max_dd):
        return 0.0
    return float(max_dd)


def sharpe_ratio(user: str = None, initial_capital: float = 100000.0) -> Optional[float]:
    df = load_trades(user=user)
    if df.empty:
        return None
    pnl = df["pnl"].astype(float)
    if len(pnl) < 2:
        return None
    mean = pnl.mean() / initial_capital
    std = pnl.std(ddof=1) / initial_capital
    if std == 0:
        return None
    annual_factor = np.sqrt(252)
    return float((mean / std) * annual_factor)


def _daily_pnl_series(user: Optional[str] = None) -> pd.Series:
    df = load_trades(user=user)
    if df.empty or "date" not in df.columns:
        return pd.Series(dtype=float)
    daily = df.groupby("date", as_index=True)["pnl"].sum().sort_index()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.astype(float)
    return daily


def cagr(user: Optional[str] = None, initial_capital: float = 100000.0) -> Optional[float]:
    daily = _daily_pnl_series(user=user)
    if daily.empty:
        return None
    start_date = daily.index.min()
    end_date = daily.index.max()
    span_days = (end_date - start_date).days
    if span_days <= 0:
        return None
    years = span_days / 365.25
    total_pnl = float(daily.cumsum().iloc[-1])
    start_value = float(initial_capital)
    end_value = start_value + total_pnl
    if start_value <= 0 or end_value <= 0:
        return None
    try:
        cagr_val = (end_value / start_value) ** (1.0 / years) - 1.0
    except Exception:
        return None
    return float(cagr_val)


def annual_volatility(user: Optional[str] = None, initial_capital: float = 100000.0) -> Optional[float]:
    daily = _daily_pnl_series(user=user)
    if daily.empty or len(daily) < 2:
        return None
    daily_std = float(daily.std(ddof=1))
    annual_vol = (daily_std * math.sqrt(252)) / float(initial_capital)
    return float(annual_vol)


def sortino_ratio(user: Optional[str] = None, initial_capital: float = 100000.0, mar: float = 0.0) -> Optional[float]:
    daily = _daily_pnl_series(user=user)
    if daily.empty or len(daily) < 2:
        return None
    diffs = daily - mar
    downside = diffs[diffs < 0]
    if len(downside) == 0:
        return None
    downside_std = float(np.sqrt((downside ** 2).mean()))
    mean_daily = float(daily.mean())
    mean_ann = (mean_daily * 252) / initial_capital
    downside_ann = (downside_std * math.sqrt(252)) / initial_capital
    if downside_ann == 0:
        return None
    return float(mean_ann / downside_ann)


def calmar_ratio(user: Optional[str] = None, initial_capital: float = 100000.0) -> Optional[float]:
    max_dd = max_drawdown(user=user)
    cagr_val = cagr(user=user, initial_capital=initial_capital)
    if cagr_val is None or max_dd is None or max_dd == 0:
        return None
    try:
        return float(cagr_val / max_dd)
    except Exception:
        return None


def expectancy_per_trade(user: Optional[str] = None) -> Optional[float]:
    df = load_trades(user=user)
    if df.empty:
        return None
    pnl = df["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate_val = len(wins) / len(pnl)
    loss_rate = 1 - win_rate_val
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    expectancy = (avg_win * win_rate_val) - (avg_loss * loss_rate)
    return float(expectancy)


def streaks(user: Optional[str] = None) -> Dict[str, int]:
    df = load_trades(user=user)
    if df.empty:
        return {"max_win_streak": 0, "max_loss_streak": 0, "current_win_streak": 0, "current_loss_streak": 0}

    pnl = df["pnl"].astype(float).tolist()
    max_win = max_loss = cur_win = cur_loss = 0
    for r in pnl:
        if r > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        if cur_win > max_win:
            max_win = cur_win
        if cur_loss > max_loss:
            max_loss = cur_loss

    last = pnl[-1]
    if last > 0:
        current_win = 1
        i = len(pnl) - 2
        while i >= 0 and pnl[i] > 0:
            current_win += 1
            i -= 1
        return {
            "max_win_streak": int(max_win),
            "max_loss_streak": int(max_loss),
            "current_win_streak": int(current_win),
            "current_loss_streak": 0,
        }
    else:
        current_loss = 1
        i = len(pnl) - 2
        while i >= 0 and pnl[i] <= 0:
            current_loss += 1
            i -= 1
        return {
            "max_win_streak": int(max_win),
            "max_loss_streak": int(max_loss),
            "current_win_streak": 0,
            "current_loss_streak": int(current_loss),
        }


def drawdown_duration(user: Optional[str] = None) -> Optional[int]:
    daily = _daily_pnl_series(user=user)
    if daily.empty:
        return None
    cum = daily.cumsum()
    running_max = cum.cummax()
    in_dd = cum < running_max
    if not in_dd.any():
        return 0
    max_dur = 0
    cur = 0
    for v in in_dd:
        if v:
            cur += 1
        else:
            if cur > max_dur:
                max_dur = cur
            cur = 0
    if cur > max_dur:
        max_dur = cur
    return int(max_dur)


def rolling_sharpe(user: Optional[str] = None, window: int = 3, initial_capital: float = 100000.0) -> Optional[pd.Series]:
    if window < 3:
        return None

    daily = _daily_pnl_series(user=user)
    if daily.empty or len(daily) < window:
        return None
    mean = daily.rolling(window=window).mean()
    std = daily.rolling(window=window).std(ddof=1)
    mean_ann = (mean * 252) / initial_capital
    std_ann = (std * math.sqrt(252)) / initial_capital
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = mean_ann / std_ann
    return rs.dropna()
'@

Write-ProjectFile "montecarlo.py" @'
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Sequence

from storage import load_trades


def _parse_targets(targets: Optional[Sequence[float]]) -> list:
    if targets is None:
        return []
    return [float(t) for t in targets]


def monte_carlo_simulation(
    user: Optional[str] = None,
    n_sims: int = 1000,
    horizon_trades: int = 252,
    seed: Optional[int] = None,
    aggregate_by_date: bool = False,
    targets: Optional[Sequence[float]] = None,
) -> Optional[Dict[str, Any]]:
    df = load_trades(user=user)
    if df.empty or "pnl" not in df.columns:
        return None

    if aggregate_by_date:
        if "date" not in df.columns:
            return None
        pnl_series = df.groupby("date", as_index=False)["pnl"].sum()["pnl"].dropna().astype(float).values
    else:
        pnl_series = df["pnl"].dropna().astype(float).values

    if len(pnl_series) == 0:
        return None

    rng = np.random.default_rng(seed)

    total_cells = int(n_sims) * int(horizon_trades)
    vectorize_threshold = 10_000_000

    if total_cells <= vectorize_threshold:
        samples = rng.choice(pnl_series, size=(int(n_sims), int(horizon_trades)), replace=True)
        cum = np.cumsum(samples, axis=1, dtype=np.float64)
    else:
        cum = np.empty((int(n_sims), int(horizon_trades)), dtype=np.float64)
        for i in range(int(n_sims)):
            sampled = rng.choice(pnl_series, size=int(horizon_trades), replace=True)
            cum[i, :] = np.cumsum(sampled, dtype=np.float64)

    endings = cum[:, -1].astype(np.float64)

    running_max = np.maximum.accumulate(cum, axis=1)
    denom = running_max.copy()
    denom[denom == 0] = np.nan
    drawdowns = (running_max - cum) / denom
    max_drawdowns = np.nanmax(drawdowns, axis=1)
    max_drawdowns = np.nan_to_num(max_drawdowns, nan=0.0, posinf=0.0, neginf=0.0)

    percentiles = dict(
        p5=float(np.percentile(endings, 5)),
        p25=float(np.percentile(endings, 25)),
        p50=float(np.percentile(endings, 50)),
        p75=float(np.percentile(endings, 75)),
        p95=float(np.percentile(endings, 95)),
    )
    prob_positive = float((endings > 0).sum() / len(endings))
    mean_terminal = float(np.mean(endings))
    median_terminal = float(np.median(endings))

    drawdown_percentiles = dict(
        p5=float(np.percentile(max_drawdowns, 5)),
        p25=float(np.percentile(max_drawdowns, 25)),
        p50=float(np.percentile(max_drawdowns, 50)),
        p75=float(np.percentile(max_drawdowns, 75)),
        p95=float(np.percentile(max_drawdowns, 95)),
    )

    targets_list = _parse_targets(targets)
    time_to_target = {}
    time_to_target_stats = {}
    prob_hit_target_map = {}

    if len(targets_list) > 0:
        for tgt in targets_list:
            if tgt >= 0:
                cond = cum >= float(tgt)
            else:
                cond = cum <= float(tgt)

            hit_any = cond.any(axis=1)
            first_hit_idx = np.argmax(cond, axis=1)
            times = np.where(hit_any, first_hit_idx + 1, np.nan).astype(np.float64)
            time_to_target[float(tgt)] = times

            prob_hit = float(hit_any.sum() / len(hit_any))
            prob_hit_target_map[float(tgt)] = prob_hit

            if hit_any.sum() > 0:
                hits_only = times[~np.isnan(times)]
                mean_time = float(np.mean(hits_only))
                median_time = float(np.median(hits_only))
                pct = dict(
                    p5=float(np.percentile(hits_only, 5)),
                    p25=float(np.percentile(hits_only, 25)),
                    p50=float(np.percentile(hits_only, 50)),
                    p75=float(np.percentile(hits_only, 75)),
                    p95=float(np.percentile(hits_only, 95)),
                )
            else:
                mean_time = None
                median_time = None
                pct = dict(p5=None, p25=None, p50=None, p75=None, p95=None)

            time_to_target_stats[float(tgt)] = {
                "prob_hit": prob_hit,
                "mean_time": mean_time,
                "median_time": median_time,
                "percentiles": pct,
                "n_hits": int(hit_any.sum()),
                "n_sims": int(len(hit_any)),
            }

    index = pd.RangeIndex(start=1, stop=int(horizon_trades) + 1, name="step")
    columns = [f"sim_{i}" for i in range(int(n_sims))]
    paths_df = pd.DataFrame(cum.T, index=index, columns=columns)

    paths_csv = paths_df.to_csv(index=True).encode("utf-8")

    dd_summary = pd.DataFrame(
        {
            "sim": [f"sim_{i}" for i in range(int(n_sims))],
            "ending": endings,
            "max_drawdown": max_drawdowns,
        }
    )
    drawdown_csv = dd_summary.to_csv(index=False).encode("utf-8")

    ttt_df = None
    ttt_csv = None
    if len(time_to_target) > 0:
        ttt_df = pd.DataFrame({"sim": [f"sim_{i}" for i in range(int(n_sims))]})
        for tgt, arr in time_to_target.items():
            col = f"time_to_target_{float(tgt)}"
            ttt_df[col] = arr
        ttt_csv = ttt_df.to_csv(index=False).encode("utf-8")

    return {
        "paths": paths_df,
        "ending": endings,
        "percentiles": percentiles,
        "prob_positive": prob_positive,
        "mean_terminal": mean_terminal,
        "median_terminal": median_terminal,
        "max_drawdowns": max_drawdowns,
        "drawdown_percentiles": drawdown_percentiles,
        "targets": targets_list,
        "time_to_target": time_to_target,
        "time_to_target_stats": time_to_target_stats,
        "prob_hit_target_map": prob_hit_target_map,
        "paths_csv": paths_csv,
        "drawdown_csv": drawdown_csv,
        "ttt_csv": ttt_csv,
        "rng_state": seed,
    }
'@

Write-ProjectFile "requirements.txt" @'
streamlit>=1.20.0
pandas>=1.5.3
numpy>=1.24.0
pytest>=7.0.0
flake8>=5.0.0
'@

Write-ProjectFile "Dockerfile" @'
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data

EXPOSE 8501

ENV PORT=8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
'@

Write-ProjectFile ".gitignore" @'
__pycache__/
*.py[cod]
*.pyo
*.pyd
.env
.venv/
venv/
pip-wheel-metadata/
.vscode/
.idea/
.DS_Store
.pytest_cache/
data/*.db
data/trades.csv
.streamlit/
'@

Write-ProjectFile ".flake8" @'
[flake8]
max-line-length = 88
extend-ignore = E203, W503
'@

Write-ProjectFile "README.md" @'
# Personal Trading System

Lightweight Streamlit app to record trades, run analytics, and run Monte Carlo simulations.

Run locally:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

Run tests:
pip install pytest
pytest -q
'@

Write-ProjectFile "tests\test_storage.py" @'
import storage
import pandas as pd

def test_save_and_load_trade(tmp_path):
    tmp_db_dir = tmp_path / "data"
    tmp_db_dir.mkdir()
    tmp_db = tmp_db_dir / "trades_test.db"
    storage.set_db_file(str(tmp_db))
    df0 = storage.load_trades(user="alice")
    assert list(df0.columns) == ["id", "user", "date", "symbol", "strategy", "pnl"]
    assert df0.empty
    row = {"date": "2023-01-01", "symbol": "AAPL", "strategy": "Test", "pnl": 12.5}
    storage.save_trade(row, user="alice")
    df = storage.load_trades(user="alice")
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "AAPL"
    assert float(df.iloc[0]["pnl"]) == 12.5
'@

Write-ProjectFile "tests\test_analytics.py" @'
import math
import storage
import analytics

def test_core_metrics_basic(tmp_path):
    tmp_db_dir = tmp_path / "data"
    tmp_db_dir.mkdir()
    tmp_db = tmp_db_dir / "trades_test.db"
    storage.set_db_file(str(tmp_db))
    storage.save_trade({"date": "2023-01-01", "symbol": "A", "strategy": "S1", "pnl": 10}, user="bob")
    storage.save_trade({"date": "2023-01-02", "symbol": "B", "strategy": "S1", "pnl": -2}, user="bob")
    storage.save_trade({"date": "2023-01-03", "symbol": "C", "strategy": "S2", "pnl": 5}, user="bob")
    win = analytics.win_rate(user="bob")
    assert math.isclose(win, 2 / 3, rel_tol=1e-9)
    expectancy = analytics.expectancy_per_trade(user="bob")
    assert abs(expectancy - 4.3333333) < 1e-6
    max_dd = analytics.max_drawdown(user="bob")
    assert abs(max_dd - 0.2) < 1e-9
    dd_dur = analytics.drawdown_duration(user="bob")
    assert dd_dur == 1
    streaks = analytics.streaks(user="bob")
    assert streaks["max_win_streak"] == 1
    assert streaks["max_loss_streak"] == 1
    cagr = analytics.cagr(user="bob", initial_capital=10000.0)
    assert cagr is None or isinstance(cagr, float)
    ann_vol = analytics.annual_volatility(user="bob", initial_capital=10000.0)
    assert ann_vol is None or (isinstance(ann_vol, float) and ann_vol >= 0.0)
    sortino = analytics.sortino_ratio(user="bob", initial_capital=10000.0)
    assert sortino is None or isinstance(sortino, float)
    calmar = analytics.calmar_ratio(user="bob", initial_capital=10000.0)
    assert calmar is None or isinstance(calmar, float)
    rolling = analytics.rolling_sharpe(user="bob", window=3, initial_capital=10000.0)
    assert rolling is None or hasattr(rolling, "shape")
'@

Write-ProjectFile "tests\test_montecarlo.py" @'
import storage
import montecarlo

def test_monte_carlo_basic(tmp_path):
    tmp_db_dir = tmp_path / "data"
    tmp_db_dir.mkdir()
    tmp_db = tmp_db_dir / "trades_test.db"
    storage.set_db_file(str(tmp_db))
    storage.save_trade({"date": "2024-01-01", "symbol": "A", "strategy": "S1", "pnl": 10}, user="mike")
    storage.save_trade({"date": "2024-01-02", "symbol": "B", "strategy": "S1", "pnl": -5}, user="mike")
    storage.save_trade({"date": "2024-01-03", "symbol": "C", "strategy": "S2", "pnl": 2}, user="mike")
    result = montecarlo.monte_carlo_simulation(user="mike", n_sims=200, horizon_trades=10, seed=42, aggregate_by_date=False, targets=[5.0])
    assert result is not None
    paths = result["paths"]
    endings = result["ending"]
    assert paths.shape[0] == 10
    assert paths.shape[1] == 200
    assert endings.shape[0] == 200
    pct = result["percentiles"]
    assert all(isinstance(v, float) for v in pct.values())
    assert 0.0 <= result["prob_positive"] <= 1.0
    assert "max_drawdowns" in result
    assert result["max_drawdowns"].shape[0] == 200
    dd_pct = result["drawdown_percentiles"]
    assert 0.0 <= dd_pct["p5"] <= 1.0
    assert 5.0 in result["time_to_target_stats"]
    assert 0.0 <= result["time_to_target_stats"][5.0]["prob_hit"] <= 1.0
    assert isinstance(result["paths_csv"], (bytes, bytearray))
    assert isinstance(result["drawdown_csv"], (bytes, bytearray))
    assert isinstance(result["ttt_csv"], (bytes, bytearray))
'@

# Create ZIP archive
if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}
Compress-Archive -Path (Join-Path $Root "*") -DestinationPath $ZipPath -Force
Write-Host "Created zip: $ZipPath"

# Create base64 file
[byte[]]$bytes = [System.IO.File]::ReadAllBytes($ZipPath)
[string]$base64 = [System.Convert]::ToBase64String($bytes)
$base64 | Out-File -FilePath $B64Path -Encoding ascii
Write-Host "Created base64 file: $B64Path"

Write-Host "Done. Files created:"
Write-Host "  $ZipPath"
Write-Host "  $B64Path"
Write-Host ""
Write-Host "To decode base64 back to zip in PowerShell:"
Write-Host "  $b = Get-Content -Raw -Path '$B64Path'"
Write-Host "  [IO.File]::WriteAllBytes('$ZipPath', [Convert]::FromBase64String($b))"
Write-Host "To unzip:"
Write-Host "  Expand-Archive -Path '$ZipPath' -DestinationPath 'path\to\extract' -Force"