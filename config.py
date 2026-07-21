"""Weekly/Daily/4H institutional options scanner configuration."""

DATA_DIR = "data"
MAX_TICKERS_PER_RUN = 80
SLEEP_BETWEEN_TICKERS = 1.0

# Kept only because the original Flask app imports it. Qualification is now
# mandatory-rule based rather than a loose score threshold.
ALERT_SCORE_MIN = 0

# Entry confirmation and risk settings
MIN_4H_RVOL = 1.30
ENTRY_BUFFER_ATR = 0.05
STOP_BUFFER_ATR = 0.10
MAX_UNDERLYING_RISK_PCT = 3.0
WATCH_DISTANCE_ATR = 1.0

WHITELIST = [
    # Index ETFs
    "SPY", "QQQ",

    # Mega Cap / AI Leaders
    "MSFT", "NVDA", "AAPL", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA",

    # Financials / Payments
    "JPM", "V", "MA", "BAC", "AFRM", "SEZL", "XYZ", "UPST", "SOFI",

    # Consumer / Retail
    "WMT", "COST", "HD", "KO", "PG",

    # Healthcare / Pharma
    "LLY", "JNJ", "ABBV", "UNH", "OSCR", "HIMS",

    # Technology / Software
    "ORCL", "CRM", "NFLX", "DDOG", "SNOW", "FROG", "TTWO",

    # Semiconductors
    "AMD", "QCOM",

    # Energy
    "XOM", "CVX",

    # Telecom
    "TMUS",

    # AI / Growth / Momentum
    "PLTR", "RGTI", "LUNR", "HOOD", "SHOP", "LMND", "RDDT",

    # Cybersecurity
    "PANW", "CRWD", "FTNT", "NET", "RBRK", "OKTA",

    # Biotech / Genomics
    "TEM", "BEAM", "TWST", "MRNA",

    # Homebuilders
    "DHI", "TOL", "PHM",

    # Crypto / Data Center / Compute
    "HUT", "MARA", "RIOT", "CLSK",

    # Misc Growth
    "GRRR",
]

BLACKLIST = []

# The scanner selects the expiration nearest the midpoint of each range.
DTE_BUCKETS = {
    "7DTE": (5, 9),
    "14DTE": (12, 16),
    "28DTE": (25, 32),
}