"""Stock universe and sector classification used in the manuscript (Table 2)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stock:
    ticker: str
    name: str
    sector: str
    data_start: int  # earliest available trading year


STOCK_UNIVERSE: list[Stock] = [
    Stock("AAPL", "Apple", "Technology", 1982),
    Stock("MSFT", "Microsoft", "Technology", 1986),
    Stock("CRM", "Salesforce", "Technology", 2004),
    Stock("JPM", "JPMorgan Chase", "Financial Services", 1982),
    Stock("V", "Visa", "Financial Services", 2008),
    Stock("JNJ", "Johnson & Johnson", "Healthcare", 1982),
    Stock("UNH", "UnitedHealth Group", "Healthcare", 1984),
    Stock("PFE", "Pfizer", "Healthcare", 1982),
    Stock("WMT", "Walmart", "Retail", 1982),
    Stock("HD", "Home Depot", "Retail", 1982),
    Stock("XOM", "ExxonMobil", "Energy", 1982),
    Stock("CVX", "Chevron", "Energy", 1982),
    Stock("PG", "Procter & Gamble", "Consumer Goods", 1982),
    Stock("KO", "Coca-Cola", "Consumer Goods", 1982),
    Stock("NKE", "Nike", "Consumer Goods", 1982),
    Stock("MCD", "McDonald's", "Consumer Goods", 1982),
    Stock("NFLX", "Netflix", "Entertainment", 2002),
    Stock("VZ", "Verizon", "Telecommunications", 1982),
    Stock("BA", "Boeing", "Industrials", 1982),
    Stock("CAT", "Caterpillar", "Industrials", 1982),
]


def ticker_to_id(ticker: str) -> int:
    for i, s in enumerate(STOCK_UNIVERSE):
        if s.ticker == ticker:
            return i
    raise KeyError(f"Unknown ticker: {ticker}")


def sector_of(ticker: str) -> str:
    for s in STOCK_UNIVERSE:
        if s.ticker == ticker:
            return s.sector
    raise KeyError(f"Unknown ticker: {ticker}")


SECTORS: list[str] = sorted({s.sector for s in STOCK_UNIVERSE})
