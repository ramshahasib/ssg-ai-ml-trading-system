# AI/ML-Based Stock Screening & Trade Filtering System

## SSG Infotech Technical Assignment

This project is a Python/Streamlit-based paper-trading prototype combining:

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

## Data note

This submission uses **demonstration/synthetic market data**. It does not connect to a live broker account and does not place real-money orders.

The purpose is to demonstrate the complete analytical, ML-filtering and paper-trading workflow without exposing broker credentials or other confidential information.

## Project files

- `app.py` — Streamlit dashboard and paper-trading logic
- `market_data.csv` — demonstration market dataset
- `model.pkl` — trained ML model used by the dashboard
- `trade_history.csv` — saved paper-trade history, when generated
- `requirements.txt` — Python dependencies
- `SSG_AI_ML_Trading_System_Report.docx` — technical report
- `README.md` — project overview

## System workflow

Market/demo data
→ feature preparation
→ SMMA 20/120
→ crossover detection
→ LTQ + Bid/Ask + price/volume analysis
→ ML probability
→ ACCEPT / AVOID
→ paper trading
→ deterioration monitoring
→ exit
→ P/L and trade history

## Running the project

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Security

No API keys, passwords, TOTP secrets or access tokens should be included in the submitted project.

All trades are simulated. No real-money orders are placed.
