import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from tej import Client

# ============================================================
# Configuration
# ============================================================

load_dotenv(Path(__file__).resolve().parent / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in your .env file.")

model = init_chat_model(
    "google_genai:gemini-3.8-flash",
    api_key=GOOGLE_API_KEY,
)

# Tej's keyless endpoint is used for daily historical NSE data.
market = Client()

# Simple local paper portfolio. This file is created on first use.
PAPER_FILE = Path("paper_portfolio.json")
INITIAL_CASH = 100_000.00  # ₹1 lakh virtual cash

# Google News RSS is a best-effort headline source, not a guaranteed API.
NEWS_TIMEOUT = 10


# ============================================================
# Market data helpers
# ============================================================

def _row_value(row, *keys):
    """Read a field from a dictionary-like Tej OHLCV row."""
    for key in keys:
        if isinstance(row, dict) and key in row:
            return row[key]
        if hasattr(row, key):
            return getattr(row, key)
    return None


def get_daily_data(ticker: str, count: int = 260):
    """
    Return up to `count` daily NSE OHLCV rows, oldest first.

    Tej SDK versions may differ in the exact response representation.
    This function normalizes common dictionary/object row formats.
    """
    ticker = ticker.upper().strip()
    if not ticker or not ticker.replace("&", "").replace("-", "").isalnum():
        raise ValueError("Use a valid NSE ticker, e.g. RELIANCE or INFY.")

    # Request a broad calendar window to obtain enough trading sessions.
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=550)).strftime("%Y-%m-%d")

    try:
        raw_rows = market.ohlcv(ticker, "nse", start_date, end_date)
    except Exception as exc:
        raise RuntimeError(f"Tej data request failed for {ticker}: {exc}") from exc

    if raw_rows is None:
        raise ValueError(f"No daily data returned for {ticker}.")

    # Some SDK versions return a DataFrame-like object.
    if hasattr(raw_rows, "to_dict") and hasattr(raw_rows, "columns"):
        rows = raw_rows.to_dict(orient="records")
    else:
        rows = list(raw_rows)

    normalized = []
    for row in rows:
        date = _row_value(row, "date", "datetime", "timestamp")
        close = _row_value(row, "close", "Close")
        if date is None or close is None:
            continue

        normalized.append({
            "date": str(date)[:10],
            "open": _row_value(row, "open", "Open"),
            "high": _row_value(row, "high", "High"),
            "low": _row_value(row, "low", "Low"),
            "close": float(close),
            "volume": _row_value(row, "volume", "Volume"),
        })

    normalized.sort(key=lambda item: item["date"])
    if not normalized:
        raise ValueError(
            f"No usable OHLCV rows for {ticker}. "
            "Check the Tej SDK response format and symbol."
        )

    return normalized[-count:]


def calculate_rsi_value(closes, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    if len(closes) < period + 1:
        raise ValueError(
            f"Need at least {period + 1} closing prices to calculate RSI."
        )

    changes = [
        closes[i] - closes[i - 1]
        for i in range(1, len(closes))
    ]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    relative_strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _latest_close(ticker: str):
    rows = get_daily_data(ticker, 2)
    latest = rows[-1]
    return latest["date"], latest["close"]


# ============================================================
# Market-analysis tools
# ============================================================

@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest available daily closing price for an NSE stock.
    Examples: RELIANCE, INFY, TCS, HDFCBANK."""
    try:
        date, price = _latest_close(ticker)
        return (
            f"{ticker.upper()} latest available daily close: ₹{price:.2f} "
            f"on {date}. This is end-of-day data, not a live quote."
        )
    except Exception as exc:
        return f"Could not fetch price for {ticker}: {exc}"


@tool
def get_price_history(ticker: str) -> str:
    """Get the last 10 available daily closing prices for an NSE stock."""
    try:
        rows = get_daily_data(ticker, 10)
        lines = [
            f"{row['date']}: ₹{row['close']:.2f}"
            for row in rows
        ]
        return f"{ticker.upper()} daily closing prices:\n" + "\n".join(lines)
    except Exception as exc:
        return f"Could not fetch history for {ticker}: {exc}"


@tool
def calculate_rsi(ticker: str) -> str:
    """Calculate the 14-period RSI from daily NSE closing prices."""
    try:
        rows = get_daily_data(ticker, 260)
        closes = [row["close"] for row in rows]
        rsi = calculate_rsi_value(closes, 14)

        if rsi > 70:
            signal = "Overbought"
        elif rsi < 30:
            signal = "Oversold"
        else:
            signal = "Neutral"

        return (
            f"{ticker.upper()} RSI(14): {rsi:.2f} — {signal}. "
            "RSI is an indicator, not a guaranteed price prediction."
        )
    except Exception as exc:
        return f"Could not calculate RSI for {ticker}: {exc}"


@tool
def get_sma(ticker: str) -> str:
    """Calculate 50-day and 200-day simple moving averages for an NSE stock."""
    try:
        rows = get_daily_data(ticker, 260)
        closes = [row["close"] for row in rows]

        if len(closes) < 200:
            return (
                f"Not enough historical prices for {ticker}'s 200-day SMA "
                f"({len(closes)} available; 200 required)."
            )

        sma_50 = sum(closes[-50:]) / 50
        sma_200 = sum(closes[-200:]) / 200

        if sma_50 > sma_200:
            alignment = "Bullish alignment: 50-day SMA is above 200-day SMA"
        elif sma_50 < sma_200:
            alignment = "Bearish alignment: 50-day SMA is below 200-day SMA"
        else:
            alignment = "Neutral alignment: both SMAs are equal"

        return (
            f"{ticker.upper()} | 50-day SMA: ₹{sma_50:.2f} | "
            f"200-day SMA: ₹{sma_200:.2f} | {alignment}. "
            "This relationship alone does not prove a crossover occurred."
        )
    except Exception as exc:
        return f"Could not calculate SMA for {ticker}: {exc}"


# ============================================================
# News tool
# ============================================================

@tool
def get_company_news(ticker: str) -> str:
    """Fetch up to five recent news headlines related to an Indian stock."""
    try:
        query = quote(f"{ticker.upper()} NSE stock India")
        url = (
            "https://news.google.com/rss/search"
            f"?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        response = requests.get(
            url,
            timeout=NEWS_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = root.findall("./channel/item")[:5]
        if not items:
            return f"No recent headlines found for {ticker}."

        lines = []
        for item in items:
            headline = item.findtext("title", "Untitled")
            published = item.findtext("pubDate", "Date unavailable")
            lines.append(f"- {headline} ({published})")

        return f"Recent news headlines for {ticker.upper()}:\n" + "\n".join(lines)
    except Exception as exc:
        return f"Could not retrieve news for {ticker}: {exc}"


# ============================================================
# Local paper-trading portfolio (no broker connection)
# ============================================================

def load_portfolio():
    if not PAPER_FILE.exists():
        portfolio = {"cash": INITIAL_CASH, "positions": {}}
        save_portfolio(portfolio)
        return portfolio

    with PAPER_FILE.open("r", encoding="utf-8") as file:
        portfolio = json.load(file)

    if "cash" not in portfolio or "positions" not in portfolio:
        raise ValueError(f"{PAPER_FILE} has an invalid portfolio format.")

    return portfolio


def save_portfolio(portfolio):
    with PAPER_FILE.open("w", encoding="utf-8") as file:
        json.dump(portfolio, file, indent=2)


@tool
def place_order(ticker: str, side: str, quantity: int) -> str:
    """Place a simulated NSE stock order at the latest available daily close.
    No real order is sent. Side must be buy or sell."""
    ticker = ticker.upper().strip()
    side = side.lower().strip()

    if side not in ("buy", "sell"):
        return "Invalid side. Use 'buy' or 'sell'."
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
        return "Quantity must be a positive whole number."

    try:
        date, price = _latest_close(ticker)
        portfolio = load_portfolio()
        positions = portfolio["positions"]
        position = positions.get(ticker, {"qty": 0, "avg_price": 0.0})
        old_qty = int(position["qty"])
        old_avg = float(position["avg_price"])
        trade_value = price * quantity

        if side == "buy":
            if trade_value > float(portfolio["cash"]):
                return (
                    f"Insufficient virtual cash. Required ₹{trade_value:.2f}; "
                    f"available ₹{float(portfolio['cash']):.2f}."
                )

            new_qty = old_qty + quantity
            new_avg = (
                (old_qty * old_avg + trade_value) / new_qty
                if new_qty else 0.0
            )
            portfolio["cash"] = round(float(portfolio["cash"]) - trade_value, 2)
            positions[ticker] = {
                "qty": new_qty,
                "avg_price": round(new_avg, 4),
            }

        else:
            if quantity > old_qty:
                return (
                    f"Cannot sell {quantity} {ticker} shares; "
                    f"you hold only {old_qty}."
                )

            portfolio["cash"] = round(float(portfolio["cash"]) + trade_value, 2)
            remaining = old_qty - quantity
            if remaining == 0:
                positions.pop(ticker, None)
            else:
                positions[ticker] = {
                    "qty": remaining,
                    "avg_price": old_avg,
                }

        save_portfolio(portfolio)
        return (
            f"SIMULATED {side.upper()} {quantity} share(s) of {ticker} "
            f"at ₹{price:.2f} using the {date} daily close. "
            f"Trade value: ₹{trade_value:.2f}. "
            f"Virtual cash remaining: ₹{portfolio['cash']:.2f}. "
            "No real order was placed."
        )
    except Exception as exc:
        return f"Paper order failed for {ticker}: {exc}"


@tool
def get_stocks_owned() -> str:
    """Show simulated NSE holdings and unrealized profit or loss."""
    try:
        portfolio = load_portfolio()
        positions = portfolio["positions"]

        if not positions:
            return "No simulated stock holdings."

        lines = ["Simulated holdings:"]
        for ticker, position in positions.items():
            _, price = _latest_close(ticker)
            quantity = int(position["qty"])
            average = float(position["avg_price"])
            pnl = (price - average) * quantity
            lines.append(
                f"{ticker}: {quantity} share(s) | "
                f"Average cost: ₹{average:.2f} | Latest close: ₹{price:.2f} | "
                f"Unrealized P/L: ₹{pnl:.2f}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Could not retrieve simulated holdings: {exc}"


@tool
def get_balance() -> str:
    """Show virtual cash, holdings value, and total portfolio value."""
    try:
        portfolio = load_portfolio()
        cash = float(portfolio["cash"])
        holdings_value = 0.0

        for ticker, position in portfolio["positions"].items():
            _, price = _latest_close(ticker)
            holdings_value += price * int(position["qty"])

        equity = cash + holdings_value
        return (
            f"Virtual cash: ₹{cash:,.2f} | "
            f"Holdings value: ₹{holdings_value:,.2f} | "
            f"Total portfolio value: ₹{equity:,.2f}"
        )
    except Exception as exc:
        return f"Could not retrieve virtual balance: {exc}"


# ============================================================
# Agent configuration
# ============================================================

SYSTEM_PROMPT = """
You are an Indian stock-market research analyst and paper-trading agent.

Market: NSE India.
Currency: Indian rupees (INR, ₹).
Price data: daily end-of-day historical data, not live quotes.

For a stock-analysis request:
1. Call get_stock_price.
2. Call get_price_history.
3. Call calculate_rsi.
4. Call get_sma.
5. Call get_company_news.
6. If relevant to an order decision, call get_stocks_owned and get_balance.
7. Evaluate the evidence and produce a report.

Use this report format:

STOCK ANALYSIS REPORT: [TICKER]
DECISION: BUY / SELL / HOLD
CONFIDENCE: High / Medium / Low

Latest available close: ₹[price] on [date]
Recent trend: [describe the observed price history]
RSI(14): [value and interpretation]
Moving averages: [50-day and 200-day SMA relationship]
News sentiment: Positive / Neutral / Negative / Unclear
Key headlines:
- [headline]
- [headline]

Rationale: Explain the evidence and uncertainty in 2–3 sentences.
Risk note: Educational paper-trading simulation only.

Rules:
- Do not invent prices, indicators, dates, headlines, or results.
- If any key market-data tool fails, disclose the missing data and do not trade.
- A 50-day SMA above the 200-day SMA is bullish alignment, not proof
  that a Golden Cross just occurred.
- RSI and moving averages are not guaranteed predictions.
- Headline sentiment can be uncertain; do not claim headlines prove future returns.
- Never describe end-of-day prices as real-time quotes.
- Never claim an order succeeded unless place_order confirms it.
- Only simulated orders are allowed.

Execution rules:
- BUY with Medium or High confidence: you may buy 1 share only when all
  required market data is available, the analysis supports the decision,
  and get_balance confirms enough virtual cash.
- SELL with High confidence: sell 1 share only if get_stocks_owned confirms
  that at least one share is held.
- HOLD: never place an order.
- Never exceed virtual cash or sell more shares than are held.
"""

agent = create_agent(
    model=model,
    tools=[
        get_stock_price,
        get_price_history,
        calculate_rsi,
        get_sma,
        get_company_news,
        place_order,
        get_stocks_owned,
        get_balance,
    ],
    system_prompt=SYSTEM_PROMPT,
)

def extract_text(content):
    """Extract readable text from an agent response."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))

        return "\n".join(parts).strip()

    return str(content)

if __name__ == "__main__":
    query = input(
        "Ask about an Indian NSE stock (e.g. RELIANCE, INFY, TCS): "
    ).strip()

    if not query:
        query = "Analyse RELIANCE on NSE India."

    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    print("\n" + result["messages"][-1].content)
