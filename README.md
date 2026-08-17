# AI/ML-Based Real-Time Stock Screening & Trade Filtering System

## SSG Infotech Technical Assignment

This project is a Python/Streamlit-based paper-trading prototype that combines:

- SMMA 20 and SMMA 120 crossover detection
- LTQ/ETQ-related features
- Bid/Ask market-depth features
- Bid/Ask imbalance
- Price and volume features
- Machine-learning probability scoring
- ACCEPT / AVOID decisions
- DETERIORATING market-condition detection
- Paper-trade entry, monitoring and exit
- Trade-history and P/L tracking

## Important note about the submitted data

This submission uses **demonstration/synthetic market data** for the technical assignment demonstration. It does not connect to a live broker account and does not place real-money orders.

The demo-data approach is intentionally used to demonstrate the complete analytical and paper-trading workflow without exposing broker credentials, API keys, access tokens or other confidential information.

## Project files

- `app.py` — Streamlit dashboard and paper-trading logic
- `market_data.csv` — demonstration market dataset
- `model.pkl` — trained ML model used by the dashboard
- `trade_history.csv` — saved paper-trade history, when generated
- `requirements.txt` — Python dependencies
- `SSG_AI_ML_Trading_System_Report.docx` — technical report
- `screenshots/` — optional screenshots for the final submission
- `README.md` — project overview

## System workflow

Market/demo data
→ feature engineering
→ SMMA 20/120
→ crossover detection
→ LTQ + Bid/Ask + price/volume features
→ ML probability
→ ACCEPT / AVOID
→ paper trade
→ deterioration monitoring
→ exit
→ P/L and trade history

## ML decision

The current demonstration dashboard uses an ML probability threshold of 0.65 for `ACCEPT`. This threshold is part of the demonstration configuration and can be optimized later with a larger out-of-sample validation dataset.

## Paper trading

All trades are simulated. No real-money order is submitted.

## Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

The application opens in the browser.

## Security

No API keys, passwords, TOTP secrets or access tokens should be included in the submitted project.
