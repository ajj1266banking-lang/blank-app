"""
import pandas as pd
import uuid

def parse_tos_csv(file_like, username):
    """
    Accepts a file-like object (uploaded CSV) and username.
    Returns a list of tuples ready for bulk_save_trades in the TRADE_COLUMNS order used by app.py.
    The ordering used in app.py is:
    id, trade_date, symbol, strategy, size, max_loss, pnl, r_multiple,
    mistakes, thesis_entry, thesis_exit, review, screenshot, user, setup, session, emotion, quality
    """
    # read via pandas
    raw = pd.read_csv(file_like)
    needed = ['Exec Time', 'Symbol', 'Net Cash']
    if not all(c in raw.columns for c in needed):
        raise ValueError("CSV doesn't look like a TOS 'Trade History' export. Expected columns: Exec Time, Symbol, Net Cash")

    raw['Date'] = pd.to_datetime(raw['Exec Time']).dt.date
    parsed = []
    for (symbol, t_date), group in raw.groupby(['Symbol', 'Date']):
        # clean Net Cash (handles strings like ($1,234.56))
        def _clean_currency(v):
            if isinstance(v, str):
                v = v.replace('$', '').replace(',', '')
                if '(' in v:
                    v = '-' + v.replace('(', '').replace(')', '')
            try:
                return float(v)
            except Exception:
                return 0.0
        pnl = group['Net Cash'].apply(_clean_currency).sum()
        # derive size if possible; best-effort fallback
        qty_sum = 1
        if 'Qty' in group.columns:
            try:
                qty_sum = int(group['Qty'].abs().sum() / 2) or 1
            except Exception:
                qty_sum = 1
        parsed.append((
            str(uuid.uuid4()), t_date.isoformat(), symbol, "TOS",
            qty_sum, 100.0, round(pnl, 2), round(pnl / 100, 2),
            "", "", "", "", "", username, "TOS Import", "Open", "Calm", 3
        ))
    return parsed
"""