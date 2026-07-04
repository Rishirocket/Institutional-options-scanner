# Options Telegram Website Bot config
DATA_DIR = "data"
MAX_TICKERS_PER_RUN = 80
SLEEP_BETWEEN_TICKERS = 1.0
OPTION_CACHE_MINUTES = 15
ALERT_SCORE_MIN = 50

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

    # Biotech / Genomics - stocks only
    "TEM", "BEAM", "TWST", "MRNA",

    # Homebuilders
    "DHI", "TOL", "PHM",

    # Crypto / Data Center / Compute
    "HUT", "MARA", "RIOT", "CLSK",

    # Misc Growth
    "GRRR",
]

BLACKLIST = []

# DTE buckets
DTE_BUCKETS = {
    "7DTE": (5, 9),
    "30DTE": (25, 40),
    "90DTE": (75, 120),
}
