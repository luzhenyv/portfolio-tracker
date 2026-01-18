CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    name TEXT,
    sector TEXT,
    industry TEXT,
    currency TEXT DEFAULT 'USD',
    exchange TEXT,
    status TEXT CHECK(status IN ('OWNED', 'WATCHLIST')) NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prices_daily (
    asset_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adjusted_close REAL,
    volume INTEGER,
    PRIMARY KEY (asset_id, date),
    FOREIGN KEY (asset_id) REFERENCES assets(id)
);
CREATE INDEX IF NOT EXISTS idx_prices_asset_date
ON prices_daily(asset_id, date);

CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
    asset_id INTEGER NOT NULL,
    quarter TEXT NOT NULL,
    revenue REAL,
    eps REAL,
    free_cash_flow REAL,
    roe REAL,
    roic REAL,
    gross_margin REAL,
    operating_margin REAL,
    debt_to_equity REAL,
    PRIMARY KEY (asset_id, quarter),
    FOREIGN KEY (asset_id) REFERENCES assets(id)
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    buy_date TEXT NOT NULL,
    shares REAL NOT NULL,
    buy_price REAL NOT NULL,
    FOREIGN KEY (asset_id) REFERENCES assets(id)
);

CREATE TABLE IF NOT EXISTS watchlist_targets (
    asset_id INTEGER PRIMARY KEY,
    fair_value REAL,
    target_buy_price REAL,
    margin_of_safety REAL,
    confidence_level TEXT CHECK(confidence_level IN ('LOW', 'MEDIUM', 'HIGH')),
    notes TEXT,
    FOREIGN KEY (asset_id) REFERENCES assets(id)
);

CREATE TABLE IF NOT EXISTS investment_thesis (
    asset_id INTEGER PRIMARY KEY,
    thesis TEXT,
    key_risks TEXT,
    red_flags TEXT,
    last_reviewed TEXT,
    FOREIGN KEY (asset_id) REFERENCES assets(id)
);

CREATE TABLE IF NOT EXISTS valuation_metrics (
    asset_id INTEGER PRIMARY KEY,
    pe_forward REAL,
    peg REAL,
    ev_ebitda REAL,
    revenue_growth REAL,
    eps_growth REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets(id)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    trade_date TEXT NOT NULL,
    action TEXT CHECK(action IN ('BUY', 'SELL', 'SHORT', 'COVER')) NOT NULL,
    shares REAL NOT NULL,
    price REAL NOT NULL,
    fees REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets(id)
);
CREATE INDEX IF NOT EXISTS idx_trades_asset_date
ON trades(asset_id, trade_date);

CREATE TABLE IF NOT EXISTS position_state (
    asset_id INTEGER PRIMARY KEY,
    long_shares REAL NOT NULL DEFAULT 0,
    long_avg_cost REAL,
    short_shares REAL NOT NULL DEFAULT 0,
    short_avg_price REAL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (asset_id) REFERENCES assets(id)
);

