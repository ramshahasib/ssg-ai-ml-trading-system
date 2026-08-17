import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="AI/ML Stock Trading System", page_icon="📈", layout="wide")

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "active_trade" not in st.session_state:
    st.session_state.active_trade = None

if "trade_history" not in st.session_state:
    st.session_state.trade_history = []

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("📈 AI/ML Real-Time Stock Screening & Trade Filtering")

st.markdown("""
**SSG Infotech Technical Assignment**

SMMA 20/120 + LTQ + Bid/Ask Market Depth + AI/ML
""")


# ---------------------------------------------------------
# Data + model loaders
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("market_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Unable to load data/model: {e}")
    st.stop()


# ---------------------------------------------------------
# Paper trading helpers
# ---------------------------------------------------------
def open_paper_trade(symbol, side, entry_price, probability, reason):
    return {
        "symbol": symbol,
        "side": side,
        "entry_price": float(entry_price),
        "entry_probability": float(probability),
        "entry_reason": reason,
        "status": "OPEN",
        "exit_price": None,
        "exit_reason": None,
        "pnl": None,
    }


def close_paper_trade(trade, exit_price, exit_reason):
    trade["exit_price"] = float(exit_price)
    trade["exit_reason"] = exit_reason

    if trade["side"] == "BUY":
        trade["pnl"] = float(exit_price) - float(trade["entry_price"])
    elif trade["side"] == "SELL":
        trade["pnl"] = float(trade["entry_price"]) - float(exit_price)

    trade["status"] = "CLOSED"
    return trade


def save_trade_to_csv(trade):
    trade_df = pd.DataFrame([trade])
    file_name = "trade_history.csv"

    try:
        existing_df = pd.read_csv(file_name)
        updated_df = pd.concat([existing_df, trade_df], ignore_index=True)
    except FileNotFoundError:
        updated_df = trade_df

    updated_df.to_csv(file_name, index=False)


# ---------------------------------------------------------
# Sidebar - stock selection
# ---------------------------------------------------------
st.sidebar.header("Stock Selection")

symbols = df["symbol"].dropna().unique()
selected_symbol = st.sidebar.selectbox("Select Stock", symbols)

stock = df[df["symbol"] == selected_symbol].copy()

if stock.empty:
    st.error("No data available for selected stock.")
    st.stop()

latest = stock.iloc[-1]

# ---------------------------------------------------------
# Top metrics
# ---------------------------------------------------------
st.subheader("Current Market Data")

col1, col2, col3, col4 = st.columns(4)
col1.metric("LTP", f"₹{latest['close']:.2f}")
col2.metric("SMMA 20", f"{latest['smma20']:.2f}")
col3.metric("SMMA 120", f"{latest['smma120']:.2f}")
col4.metric("LTQ", f"{latest['ltq']:,.0f}")

# ---------------------------------------------------------
# Market depth
# ---------------------------------------------------------
st.subheader("Market Depth")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Bid Price", f"₹{latest['bid_price']:.2f}")
col2.metric("Bid Quantity", f"{latest['bid_qty']:,.0f}")
col3.metric("Ask Price", f"₹{latest['ask_price']:.2f}")
col4.metric("Ask Quantity", f"{latest['ask_qty']:,.0f}")

# bid/ask imbalance
bid_qty = float(latest["bid_qty"])
ask_qty = float(latest["ask_qty"])

if (bid_qty + ask_qty) != 0:
    imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
else:
    imbalance = 0

st.metric("Bid/Ask Imbalance", f"{imbalance:.2%}")

# ---------------------------------------------------------
# Price / SMMA chart
# ---------------------------------------------------------
st.subheader("Price & SMMA")

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock["timestamp"], y=stock["close"], name="LTP"))
fig.add_trace(go.Scatter(x=stock["timestamp"], y=stock["smma20"], name="SMMA 20"))
fig.add_trace(go.Scatter(x=stock["timestamp"], y=stock["smma120"], name="SMMA 120"))
fig.update_layout(height=500, xaxis_title="Time", yaxis_title="Price")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Crossover signal
# ---------------------------------------------------------
latest_signal = str(latest["signal"])

st.subheader("Trading Signal")

if latest_signal == "BUY":
    st.success("🟢 BUY crossover detected")
elif latest_signal == "SELL":
    st.error("🔴 SELL crossover detected")
else:
    st.info("No crossover at current candle")

# ---------------------------------------------------------
# ML features + prediction
# ---------------------------------------------------------
FEATURES = [
    "ltq_change",
    "ltq_acceleration",
    "bid_qty_change",
    "ask_qty_change",
    "bid_ask_imbalance",
    "bid_ask_spread",
    "spread_pct",
    "price_change",
    "price_momentum_5",
    "price_momentum_10",
    "volume_change",
    "relative_volume",
    "smma_spread",
    "smma_spread_change",
    "price_vs_smma20",
]

latest_row = stock.iloc[-1]

missing_features = [f for f in FEATURES if f not in stock.columns]
if missing_features:
    st.error("Missing ML features: " + ", ".join(missing_features))
    st.stop()

X = pd.DataFrame([latest_row[FEATURES]])
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

try:
    probability = model.predict_proba(X)[0][1]
except Exception as e:
    st.error(f"ML prediction failed: {e}")
    st.stop()

# 0.65 is currently being used for testing.
# We'll tune this threshold later once we have Day-2 validation data.
decision = "ACCEPT" if probability >= 0.65 else "AVOID"

st.subheader("AI/ML Decision")

col1, col2 = st.columns(2)
col1.metric("ML Probability", f"{probability:.2%}")

if decision == "ACCEPT":
    col2.success("✅ ACCEPT")
else:
    col2.warning("⚠️ AVOID")

# ---------------------------------------------------------
# Reasons behind the decision
# ---------------------------------------------------------
reasons = []

if latest_row["ltq_change"] > 0:
    reasons.append("LTQ increasing")
else:
    reasons.append("LTQ weakening")

if imbalance > 0.20:
    reasons.append("Strong bid support")
elif imbalance < -0.20:
    reasons.append("Strong ask pressure")
else:
    reasons.append("Neutral Bid/Ask imbalance")

if latest_row["smma_spread_change"] > 0:
    reasons.append("SMMA spread strengthening")
else:
    reasons.append("SMMA spread weakening")

trade_reason = "; ".join(reasons)

st.subheader("Decision Reasons")
for reason in reasons:
    st.write("• " + reason)

# ---------------------------------------------------------
# Deterioration checks
# ---------------------------------------------------------
deterioration_reasons = []
signal = latest_signal

if signal == "BUY":
    if latest_row["ltq_change"] < 0:
        deterioration_reasons.append("LTQ is declining after BUY signal")
    if latest_row["bid_qty_change"] < -0.10:
        deterioration_reasons.append("Bid support is weakening")
    if latest_row["ask_qty_change"] > 0.10:
        deterioration_reasons.append("Ask pressure is increasing")
    if imbalance < -0.20:
        deterioration_reasons.append("Bid/Ask imbalance turned negative")
    if latest_row["smma_spread_change"] < 0:
        deterioration_reasons.append("SMMA spread is weakening")
    if latest_row["price_change"] < 0:
        deterioration_reasons.append("Price is moving against BUY position")

elif signal == "SELL":
    if latest_row["ltq_change"] < 0:
        deterioration_reasons.append("LTQ is declining after SELL signal")
    if latest_row["bid_qty_change"] > 0.10:
        deterioration_reasons.append("Bid pressure is increasing")
    if latest_row["ask_qty_change"] < -0.10:
        deterioration_reasons.append("Ask-side support is weakening")
    if imbalance > 0.20:
        deterioration_reasons.append("Bid/Ask imbalance turned positive")
    if latest_row["smma_spread_change"] > 0:
        deterioration_reasons.append("SMMA spread is moving against SELL")
    if latest_row["price_change"] > 0:
        deterioration_reasons.append("Price is moving against SELL")

if len(deterioration_reasons) >= 3:
    market_status = "DETERIORATING"
elif len(deterioration_reasons) >= 1:
    market_status = "WARNING"
else:
    market_status = "STABLE"

st.subheader("Market Condition")

if market_status == "DETERIORATING":
    st.error("🔴 DETERIORATING")
    st.write("Multiple market conditions are moving against the current signal.")
    for reason in deterioration_reasons:
        st.write("• " + reason)

elif market_status == "WARNING":
    st.warning("🟡 WARNING")
    st.write("Early signs of deterioration detected.")
    for reason in deterioration_reasons:
        st.write("• " + reason)

else:
    st.success("🟢 STABLE")
    st.write("Market conditions remain favorable.")

# ---------------------------------------------------------
# Paper trading
# ---------------------------------------------------------
st.subheader("Paper Trading")

# open a new trade if we get a clean ACCEPT + signal and nothing's open
if (
    decision == "ACCEPT"
    and latest_signal in ["BUY", "SELL"]
    and st.session_state.active_trade is None
):
    st.session_state.active_trade = open_paper_trade(
        symbol=selected_symbol,
        side=latest_signal,
        entry_price=latest["close"],
        probability=probability,
        reason=trade_reason,
    )
    st.success("🟢 Paper trade opened")

trade = st.session_state.active_trade

if trade is not None:
    st.write("### Active Paper Trade")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Direction", trade["side"])
    col2.metric("Entry Price", f"₹{trade['entry_price']:.2f}")
    col3.metric("Current Price", f"₹{latest['close']:.2f}")

    if trade["side"] == "BUY":
        current_pnl = latest["close"] - trade["entry_price"]
    else:
        current_pnl = trade["entry_price"] - latest["close"]

    col4.metric("Current P/L", f"₹{current_pnl:.2f}")

    # auto-close if things are deteriorating
    if market_status == "DETERIORATING":
        closed_trade = close_paper_trade(trade, latest["close"], "Market conditions deteriorated")
        st.session_state.trade_history.append(closed_trade)
        save_trade_to_csv(closed_trade)
        st.session_state.active_trade = None

        st.error("🔴 Paper trade automatically closed because market conditions deteriorated.")
    else:
        st.info("Paper trade is currently being monitored.")

    # manual close button
    if st.session_state.active_trade is not None:
        st.write("### Manual Trade Control")

        if st.button("🔴 Close Paper Trade"):
            trade_to_close = st.session_state.active_trade
            closed_trade = close_paper_trade(trade_to_close, latest["close"], "Manual paper exit")

            st.session_state.trade_history.append(closed_trade)
            save_trade_to_csv(closed_trade)
            st.session_state.active_trade = None

            st.success("✅ Paper trade closed successfully.")
            st.rerun()

# ---------------------------------------------------------
# Trade history
# ---------------------------------------------------------
st.subheader("Paper Trade History")

if len(st.session_state.trade_history) > 0:
    history_df = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No completed paper trades yet.")

st.subheader("Saved Trade History")

try:
    saved_trades = pd.read_csv("trade_history.csv")
    st.dataframe(saved_trades, use_container_width=True)
except FileNotFoundError:
    st.info("No saved trades yet.")

# ---------------------------------------------------------
# Performance summary
# ---------------------------------------------------------
if len(st.session_state.trade_history) > 0:
    history_df = pd.DataFrame(st.session_state.trade_history)

    total_trades = len(history_df)
    winning_trades = (history_df["pnl"] > 0).sum()
    losing_trades = (history_df["pnl"] <= 0).sum()
    total_pnl = history_df["pnl"].sum()

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    st.subheader("Paper Trading Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", total_trades)
    col2.metric("Winning Trades", winning_trades)
    col3.metric("Losing Trades", losing_trades)
    col4.metric("Win Rate", f"{win_rate:.2f}%")

    st.metric("Total Paper P/L", f"₹{total_pnl:.2f}")

# ---------------------------------------------------------
# Recent market data table
# ---------------------------------------------------------
st.subheader("Recent Market Data")

display_columns = [
    "timestamp",
    "symbol",
    "close",
    "smma20",
    "smma120",
    "ltq",
    "bid_qty",
    "ask_qty",
    "bid_ask_imbalance",
    "signal",
]

available_display_columns = [c for c in display_columns if c in stock.columns]

st.dataframe(stock[available_display_columns].tail(20), use_container_width=True)

# ---------------------------------------------------------
# System info footer
# ---------------------------------------------------------
st.subheader("System Information")

col1, col2, col3 = st.columns(3)
col1.metric("Selected Stock", selected_symbol)
col2.metric("Current Signal", latest_signal)
col3.metric("AI Decision", decision)

st.caption("Paper trading only. No real-money orders are placed.")
