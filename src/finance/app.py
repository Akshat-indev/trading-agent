import streamlit as st
from datetime import datetime

from trading_agent import (
    agent,
    extract_text,
    load_portfolio,
    _latest_close,
    INITIAL_CASH,
)

st.set_page_config(
    page_title="NSE Paper Trader",
    page_icon="📈",
    layout="wide",
)

st.title("NSE Paper Trader")
st.caption("AI-powered stock research and simulated trading · NSE India · INR")

st.warning(
    "Paper trading only. Prices are the latest available daily closes, not live quotes. "
    "No real brokerage orders are placed."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "last_portfolio_refresh" not in st.session_state:
    st.session_state.last_portfolio_refresh = None


def portfolio_snapshot():
    portfolio = load_portfolio()
    cash = float(portfolio.get("cash", INITIAL_CASH))
    holdings_value = 0.0
    unrealized_pnl = 0.0
    rows = []
    errors = []

    for ticker, position in portfolio.get("positions", {}).items():
        qty = int(position.get("qty", 0))
        avg_price = float(position.get("avg_price", 0.0))
        if qty <= 0:
            continue
        try:
            date, latest_price = _latest_close(ticker)
            market_value = qty * latest_price
            pnl = (latest_price - avg_price) * qty
            holdings_value += market_value
            unrealized_pnl += pnl
            rows.append({
                "Ticker": ticker,
                "Quantity": qty,
                "Avg. cost (₹)": round(avg_price, 2),
                "Latest close (₹)": round(latest_price, 2),
                "Market value (₹)": round(market_value, 2),
                "Unrealized P/L (₹)": round(pnl, 2),
                "Price date": date,
            })
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
            rows.append({
                "Ticker": ticker,
                "Quantity": qty,
                "Avg. cost (₹)": round(avg_price, 2),
                "Latest close (₹)": None,
                "Market value (₹)": None,
                "Unrealized P/L (₹)": None,
                "Price date": "Unavailable",
            })

    return {
        "cash": cash,
        "holdings_value": holdings_value,
        "total_value": cash + holdings_value,
        "unrealized_pnl": unrealized_pnl,
        "rows": rows,
        "errors": errors,
    }


def show_portfolio():
    try:
        snapshot = portfolio_snapshot()
    except Exception as exc:
        st.error(f"Could not load portfolio: {exc}")
        return

    a, b, c, d = st.columns(4)
    a.metric("Virtual cash", f"₹{snapshot['cash']:,.2f}")
    b.metric("Holdings value", f"₹{snapshot['holdings_value']:,.2f}")
    c.metric("Total portfolio", f"₹{snapshot['total_value']:,.2f}")
    d.metric("Unrealized P/L", f"₹{snapshot['unrealized_pnl']:,.2f}")

    if snapshot["rows"]:
        st.dataframe(snapshot["rows"], use_container_width=True, hide_index=True)
    else:
        st.info("No holdings yet. Ask for analysis first, then explicitly instruct the agent to place a simulated order.")

    if snapshot["errors"]:
        st.caption("Some holding prices could not be refreshed: " + " | ".join(snapshot["errors"]))


with st.sidebar:
    st.subheader("Portfolio")
    if st.button("Refresh portfolio", use_container_width=True):
        st.session_state.last_portfolio_refresh = datetime.now().strftime("%H:%M:%S")
        st.rerun()
    show_portfolio()
    if st.session_state.last_portfolio_refresh:
        st.caption(f"Last refreshed: {st.session_state.last_portfolio_refresh}")
    st.divider()
    st.subheader("Try asking")
    examples = [
        "Analyse TCS and give me a BUY/SELL/HOLD recommendation. Do not trade.",
        "What is my current balance and total portfolio value?",
        "Show my current NSE holdings and unrealized P/L.",
        "Compare INFY and TCS using recent price history, RSI, moving averages, and news.",
        "Buy 2 shares of TCS in my paper portfolio.",
        "Sell 1 share of INFY from my paper portfolio.",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_prompt = example

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about stocks, your portfolio, or a simulated trade…")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking market data and thinking…"):
            try:
                next_messages = list(st.session_state.agent_messages)
                next_messages.append({"role": "user", "content": prompt})
                result = agent.invoke({"messages": next_messages})
                st.session_state.agent_messages = result["messages"]
                answer = extract_text(result["messages"][-1].content)
                if not answer:
                    answer = "The agent returned no text. Check the terminal for tool or API errors."
            except Exception as exc:
                answer = (
                    f"Agent error: `{exc}`\n\n"
                    "Check your `.env` file, API key, installed packages, model name, "
                    "and Tej SDK response format."
                )
            st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()
