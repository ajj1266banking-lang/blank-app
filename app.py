import streamlit as st
import pandas as pd
import sqlite3
import uuid
import hashlib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, timedelta
from pathlib import Path

# ---------------- CONFIG & DIRECTORIES ----------------
DB_PATH = Path("trades.db")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

STRATEGIES = ["Call", "Put", "Credit Spread", "Calendar", "Diagonal", "Butterfly", "Stock", "Vertical"]
MISTAKES = ["Chased entry", "Ignored stop", "Oversized", "Traded outside plan", "Revenge trade", "FOMO", "Held too long"]
SESSIONS = ["Open", "Midday", "Close"]
EMOTIONS = ["Calm", "Confident", "FOMO", "Revenge", "Fear", "Tired"]

# Risk settings
MAX_DRAWDOWN_WARN = -500
MAX_DRAWDOWN_STOP = -1000
MAX_TRADES_PER_DAY = 3

st.set_page_config(page_title="Elite Trader System", layout="wide")

# canonical trade column order (must match the DB trades table order)
TRADE_COLUMNS = [
    "id", "trade_date", "symbol", "strategy",
    "size", "max_loss", "pnl", "r_multiple",
    "mistakes", "thesis_entry", "thesis_exit", "review",
    "screenshot", "user", "setup", "session",
    "emotion", "quality"
]

# ---------------- DATABASE LOGIC ----------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS trades (
                {', '.join([
                    'id TEXT PRIMARY KEY',
                    'trade_date TEXT',
                    'symbol TEXT',
                    'strategy TEXT',
                    'size INTEGER',
                    'max_loss REAL',
                    'pnl REAL',
                    'r_multiple REAL',
                    'mistakes TEXT',
                    'thesis_entry TEXT',
                    'thesis_exit TEXT',
                    'review TEXT',
                    'screenshot TEXT',
                    'user TEXT',
                    'setup TEXT',
                    'session TEXT',
                    'emotion TEXT',
                    'quality INTEGER'
                ])}
            )
        """
        )

def save_trade(trade, edit=False):
    """
    trade: dict mapping TRADE_COLUMNS -> values
    """
    with get_conn() as conn:
        if edit:
            # build SET clause and parameters in the canonical order (excluding id)
            cols_no_id = TRADE_COLUMNS[1:]
            set_clause = ", ".join([f"{c}=?" for c in cols_no_id])
            params = tuple(trade.get(c) for c in cols_no_id) + (trade["id"],)
            conn.execute(f"UPDATE trades SET {set_clause} WHERE id=?", params[:-1] + (params[-1],))
        else:
            placeholders = ",".join(["?" ] * len(TRADE_COLUMNS))
            params = tuple(trade.get(c) for c in TRADE_COLUMNS)
            conn.execute(f"INSERT INTO trades VALUES ({placeholders})", params)

def bulk_save_trades(trades_list):
    """
    trades_list: iterable of tuples/lists already in the TRADE_COLUMNS order (length must match)
    """
    placeholders = ",".join(["?" ] * len(TRADE_COLUMNS))
    with get_conn() as conn:
        conn.executemany(f"INSERT INTO trades VALUES ({placeholders})", trades_list)

init_db()

# ---------------- AUTH & SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    st.title("Elite Journal Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login/Register"):
        if not u or not p:
            st.error("Please provide both username and password.")
        else:
            pw_hash = hashlib.sha256(p.encode()).hexdigest()
            with get_conn() as conn:
                # Insert the user if they don't exist yet
                conn.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?,?)", (u, pw_hash))
                res = conn.execute("SELECT password FROM users WHERE username=?", (u,)).fetchone()
                if res is None:
                    st.error("Unexpected error: user record not found after insert.")
                else:
                    stored_pw = res[0]
                    if stored_pw == pw_hash:
                        st.session_state.user = u
                        st.rerun()
                    else:
                        st.error("Incorrect password for that username. If this is the first time you use this app, please register by entering a new username and password and pressing the button again.")

    st.stop()

# ---------------- DATA LOADING & HELPERS ----------------
# load current user's trades
conn = get_conn()
df = pd.read_sql("SELECT * FROM trades WHERE user = ?", conn, params=(st.session_state.user,))
if not df.empty and 'trade_date' in df.columns:
    df['trade_date'] = pd.to_datetime(df['trade_date'])

def clean_currency(value):
    if isinstance(value, str):
        value = value.replace('$', '').replace(',', '')
        if '(' in value:
            value = '-' + value.replace('(', '').replace(')', '')
    try:
        return float(value)
    except Exception:
        return 0.0

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title(f"Welcome, {st.session_state.user}")
    page = st.radio("Navigate", ["Dashboard", "Add / Edit Trade", "TOS Import", "Analytics", "All Trades"]) 
    
    # Live Risk Status
    if not df.empty:
        equity = df.sort_values("trade_date").pnl.cumsum()
        max_dd = (equity - equity.cummax()).min()
        st.metric("Max Drawdown", f"${max_dd:.2f}")
        progress_val = min(abs(max_dd) / abs(MAX_DRAWDOWN_STOP), 1.0) if max_dd < 0 else 0.0
        st.progress(progress_val)
    
    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.header("Dashboard")
    if df.empty:
        st.info("No trades yet. Upload a TOS CSV or add one manually.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total P/L", f"${df.pnl.sum():,.2f}")
        c2.metric("Win Rate", f"{(df.pnl > 0).mean()*100:.1f}%")
        c3.metric("Avg R", f"{df.r_multiple.mean():.2f}")
        c4.metric("Trades", len(df))
        st.line_chart(df.sort_values("trade_date").set_index("trade_date")['pnl'].cumsum())

# ---------------- TOS IMPORT ----------------
elif page == "TOS Import":
    st.header("ThinkOrSwim Import")
    file = st.file_uploader("Upload 'Trade History' CSV from TOS", type="csv")
    if file:
        raw_tos = pd.read_csv(file)
        if st.button("Process & Save"):
            parsed = []
            # Grouping by Symbol and Date to combine executions
            if 'Exec Time' not in raw_tos.columns or 'Symbol' not in raw_tos.columns or 'Net Cash' not in raw_tos.columns:
                st.error("CSV doesn't look like a TOS 'Trade History' export. Expected columns like 'Exec Time', 'Symbol', 'Net Cash'.")
            else:
                raw_tos['Date'] = pd.to_datetime(raw_tos['Exec Time']).dt.date
                for (symbol, t_date), group in raw_tos.groupby(['Symbol', 'Date']):
                    pnl = group['Net Cash'].apply(clean_currency).sum()
                    # Ensure qty exists and is numeric; fall back to 1 if not present
                    qty_sum = 0
                    if 'Qty' in group.columns:
                        try:
                            qty_sum = int(group['Qty'].abs().sum() / 2)
                        except Exception:
                            qty_sum = 1
                    else:
                        qty_sum = 1
                    parsed.append((
                        str(uuid.uuid4()), t_date.isoformat(), symbol, "TOS",
                        qty_sum, 100.0, round(pnl, 2), round(pnl / 100, 2),
                        "", "", "", "", "", st.session_state.user, "TOS Import", "Open", "Calm", 3
                    ))
                if parsed:
                    bulk_save_trades(parsed)
                st.success(f"Imported {len(parsed)} trades.")
                st.balloons()

# ---------------- ANALYTICS & HEATMAP ----------------
elif page == "Analytics":
    st.header("Performance Analytics")
    if df.empty:
        st.info("No trades to analyze.")
    else:
        df['Day'] = df['trade_date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # pivot on 'session' column, which matches the DB column name
        heatmap_data = df.pivot_table(index='session', columns='Day', values='pnl', aggfunc='sum').reindex(columns=day_order).fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="RdYlGn", center=0, ax=ax)
        st.pyplot(fig)
        
        # Strategy Stats
        st.subheader("Strategy Performance")
        st.dataframe(df.groupby('strategy')['pnl'].agg(['count', 'sum', 'mean']))

# ---------------- ADD / EDIT ----------------
elif page == "Add / Edit Trade":
    st.header("Manual Entry")
    with st.form("manual_form"):
        t1, t2 = st.tabs(["Data", "Psychology"])
        with t1:
            d = st.date_input("Date")
            s = st.text_input("Symbol").upper()
            p = st.number_input("P/L", value=0.0)
            r = st.number_input("Risk ($)", value=100.0)
            size = st.number_input("Size", value=1)
        with t2:
            m = st.multiselect("Mistakes", MISTAKES)
            sess = st.selectbox("Session", SESSIONS)
            emo = st.selectbox("Emotion", EMOTIONS)
            review = st.text_area("Review Notes")
        
        if st.form_submit_button("Save Trade"):
            r_mult = p / r if r > 0 else 0
            new_t = {
                "id": str(uuid.uuid4()),
                "trade_date": d.isoformat(),
                "symbol": s,
                "strategy": "Manual",
                "size": int(size),
                "max_loss": float(r),
                "pnl": float(p),
                "r_multiple": float(r_mult),
                "mistakes": ",".join(m),
                "thesis_entry": "",
                "thesis_exit": "",
                "review": review,
                "screenshot": "",
                "user": st.session_state.user,
                "setup": "Manual",
                "session": sess,
                "emotion": emo,
                "quality": 3
            }
            save_trade(new_t, edit=False)
            st.success("Saved!")

# ---------------- ALL TRADES ----------------
elif page == "All Trades":
    st.header("Trade History")
    if df.empty:
        st.info("No trades yet.")
    else:
        st.dataframe(df.sort_values("trade_date", ascending=False))
