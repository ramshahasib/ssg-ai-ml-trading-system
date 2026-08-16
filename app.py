import time
REFRESH_SECONDS = 10
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
# --------------------------------------------------
# PAPER TRADING STATE
# --------------------------------------------------

if "active_trade" not in st.session_state:

    st.session_state.active_trade = None


if "trade_history" not in st.session_state:

    st.session_state.trade_history = []

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI/ML Stock Trading System",
    page_icon="📈",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📈 AI/ML Real-Time Stock Screening & Trade Filtering")

st.markdown(
    """
    **SSG Infotech Technical Assignment**

    SMMA 20/120 + LTQ + Bid/Ask Market Depth + AI/ML
    """
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "market_data.csv"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


@st.cache_resource
def load_model():

    return joblib.load(
        "model.pkl"
    )


df = load_data()

model = load_model()

# --------------------------------------------------
# PAPER TRADE FUNCTION
# --------------------------------------------------

def open_paper_trade(
    symbol,
    side,
    entry_price,
    probability,
    reason
):

    return {

        "symbol": symbol,

        "side": side,

        "entry_price": entry_price,

        "entry_probability": probability,

        "entry_reason": reason,

        "status": "OPEN",

        "exit_price": None,

        "exit_reason": None,

        "pnl": None

    }
# --------------------------------------------------
# CLOSE PAPER TRADE
# --------------------------------------------------

def close_paper_trade(
    trade,
    exit_price,
    exit_reason
):

    trade["exit_price"] = exit_price

    trade["exit_reason"] = exit_reason

    if trade["side"] == "BUY":

        trade["pnl"] = (
            exit_price -
            trade["entry_price"]
        )

    elif trade["side"] == "SELL":

        trade["pnl"] = (
            trade["entry_price"] -
            exit_price
        )

    trade["status"] = "CLOSED"

    return trade



# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header(
    "Stock Selection"
)

symbols = df[
    "symbol"
].unique()

selected_symbol = st.sidebar.selectbox(
    "Select Stock",
    symbols
)


stock = df[
    df["symbol"] == selected_symbol
].copy()


# --------------------------------------------------
# LATEST DATA
# --------------------------------------------------

latest = stock.iloc[-1]


# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "LTP",
    f"₹{latest['close']:.2f}"
)

col2.metric(
    "SMMA 20",
    f"{latest['smma20']:.2f}"
)

col3.metric(
    "SMMA 120",
    f"{latest['smma120']:.2f}"
)

col4.metric(
    "LTQ",
    f"{latest['ltq']:,.0f}"
)


# --------------------------------------------------
# MARKET DEPTH
# --------------------------------------------------

st.subheader(
    "Market Depth"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Bid Price",
    f"₹{latest['bid_price']:.2f}"
)

col2.metric(
    "Bid Quantity",
    f"{latest['bid_qty']:,.0f}"
)

col3.metric(
    "Ask Price",
    f"₹{latest['ask_price']:.2f}"
)

col4.metric(
    "Ask Quantity",
    f"{latest['ask_qty']:,.0f}"
)


# --------------------------------------------------
# BID ASK IMBALANCE
# --------------------------------------------------

imbalance = (
    latest["bid_qty"] -
    latest["ask_qty"]
) / (
    latest["bid_qty"] +
    latest["ask_qty"]
)


st.metric(
    "Bid/Ask Imbalance",
    f"{imbalance:.2%}"
)


# --------------------------------------------------
# SMMA CHART
# --------------------------------------------------

st.subheader(
    "Price & SMMA"
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=stock["timestamp"],
        y=stock["close"],
        name="LTP"
    )
)

fig.add_trace(
    go.Scatter(
        x=stock["timestamp"],
        y=stock["smma20"],
        name="SMMA 20"
    )
)

fig.add_trace(
    go.Scatter(
        x=stock["timestamp"],
        y=stock["smma120"],
        name="SMMA 120"
    )
)

fig.update_layout(
    height=500,
    xaxis_title="Time",
    yaxis_title="Price"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# LATEST SIGNAL
# --------------------------------------------------

latest_signal = latest["signal"]

st.subheader(
    "Trading Signal"
)

if latest_signal == "BUY":

    st.success(
        "🟢 BUY crossover detected"
    )

elif latest_signal == "SELL":

    st.error(
        "🔴 SELL crossover detected"
    )

else:

    st.info(
        "No crossover at current candle"
    )


# --------------------------------------------------
# ML PREDICTION
# --------------------------------------------------

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

    "price_vs_smma20"

]


latest_row = stock.iloc[-1]

X = pd.DataFrame(
    [latest_row[FEATURES]]
)


X = X.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)


probability = model.predict_proba(
    X
)[0][1]


# --------------------------------------------------
# DECISION
# --------------------------------------------------

if probability >= 0.75:

    decision = "ACCEPT"

elif probability >= 0.50:

    decision = "AVOID"

else:

    decision = "AVOID"


st.subheader(
    "AI/ML Decision"
)

col1, col2 = st.columns(2)

col1.metric(
    "ML Probability",
    f"{probability:.2%}"
)

if decision == "ACCEPT":

    col2.success(
        f"✅ {decision}"
    )

else:

    col2.warning(
        f"⚠️ {decision}"
    )


# --------------------------------------------------
# REASON
# --------------------------------------------------

reasons = []


if latest_row["ltq_change"] > 0:

    reasons.append(
        "LTQ increasing"
    )

else:

    reasons.append(
        "LTQ weakening"
    )


if imbalance > 0.20:

    reasons.append(
        "Strong bid support"
    )

elif imbalance < -0.20:

    reasons.append(
        "Strong ask pressure"
    )

else:

    reasons.append(
        "Neutral Bid/Ask imbalance"
    )


if latest_row["smma_spread_change"] > 0:

    reasons.append(
        "SMMA spread strengthening"
    )

else:

    reasons.append(
        "SMMA spread weakening"
    )


st.subheader(
    "Decision Reasons"
)

for reason in reasons:

    st.write(
        "• " + reason
    )
    trade_reason = "; ".join(
    reasons
)
# --------------------------------------------------
# SMART DETERIORATION DETECTION
# --------------------------------------------------

deterioration_reasons = []

signal = latest_row["signal"]


# ================================================
# BUY TRADE
# ================================================

if signal == "BUY":

    # LTQ falling
    if latest_row["ltq_change"] < 0:

        deterioration_reasons.append(
            "LTQ is declining after BUY signal"
        )

    # Bid support weakening
    if latest_row["bid_qty_change"] < -0.10:

        deterioration_reasons.append(
            "Bid support is weakening"
        )

    # Ask pressure increasing
    if latest_row["ask_qty_change"] > 0.10:

        deterioration_reasons.append(
            "Ask pressure is increasing"
        )

    # Negative imbalance
    if imbalance < -0.20:

        deterioration_reasons.append(
            "Bid/Ask imbalance turned negative"
        )

    # SMMA weakening
    if latest_row["smma_spread_change"] < 0:

        deterioration_reasons.append(
            "SMMA spread is weakening"
        )

    # Price falling
    if latest_row["price_change"] < 0:

        deterioration_reasons.append(
            "Price is moving against BUY position"
        )


# ================================================
# SELL TRADE
# ================================================

elif signal == "SELL":

    # LTQ falling can indicate weakening selling activity
    if latest_row["ltq_change"] < 0:

        deterioration_reasons.append(
            "LTQ is declining after SELL signal"
        )

    # Bid pressure increasing
    if latest_row["bid_qty_change"] > 0.10:

        deterioration_reasons.append(
            "Bid pressure is increasing"
        )

    # Ask support weakening
    if latest_row["ask_qty_change"] < -0.10:

        deterioration_reasons.append(
            "Ask-side support is weakening"
        )

    # Positive imbalance
    if imbalance > 0.20:

        deterioration_reasons.append(
            "Bid/Ask imbalance turned positive"
        )

    # SMMA weakening for SELL
    if latest_row["smma_spread_change"] > 0:

        deterioration_reasons.append(
            "SMMA spread is moving against SELL"
        )

    # Price rising
    if latest_row["price_change"] > 0:

        deterioration_reasons.append(
            "Price is moving against SELL position"
        )


# ================================================
# MARKET STATUS
# ================================================

if len(deterioration_reasons) >= 3:

    market_status = "DETERIORATING"

elif len(deterioration_reasons) >= 1:

    market_status = "WARNING"

else:

    market_status = "STABLE"


# ================================================
# DISPLAY
# ================================================

st.subheader(
    "Market Condition"
)


if market_status == "DETERIORATING":

    st.error(
        "🔴 DETERIORATING"
    )

    st.write(
        "Multiple market conditions are moving "
        "against the current signal."
    )

    for reason in deterioration_reasons:

        st.write(
            "• " + reason
        )


elif market_status == "WARNING":

    st.warning(
        "🟡 WARNING"
    )

    st.write(
        "Early signs of deterioration detected."
    )

    for reason in deterioration_reasons:

        st.write(
            "• " + reason
        )


else:

    st.success(
        "🟢 STABLE"
    )

    st.write(
        "Market conditions remain favorable."
    )
# --------------------------------------------------
# PAPER TRADING
# --------------------------------------------------

st.subheader(
    "Paper Trading"
)


if decision == "ACCEPT":

    if market_status == "DETERIORATING":

        st.error(
            "🚨 PAPER TRADE EXIT"
        )

        st.write(
            f"Direction: {latest_signal}"
        )

        st.write(
            f"Entry Price: ₹{latest['close']:.2f}"
        )

        st.write(
            "Reason: Market conditions deteriorated."
        )

        st.write(
            "**Deterioration factors:**"
        )

        for reason in deterioration_reasons:

            st.write(
                "• " + reason
            )

    else:

        st.success(
            "🟢 PAPER TRADE OPEN"
        )

        st.write(
            f"Direction: {latest_signal}"
        )

        st.write(
            f"Entry Price: ₹{latest['close']:.2f}"
        )

        st.write(
            f"AI Probability: {probability:.2%}"
        )

        st.write(
            "Status: Monitoring"
        )


else:

    st.warning(
        "No paper trade opened."
    )

    st.write(
        "The AI/ML filter rejected this signal."
    )
# --------------------------------------------------
# RECENT DATA
# --------------------------------------------------

st.subheader(
    "Recent Market Data"
)

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

    "signal"

]

st.dataframe(
    stock[
        display_columns
    ].tail(20),
    use_container_width=True
)
time.sleep(
    REFRESH_SECONDS
)

st.rerun()
