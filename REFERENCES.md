# References

## Data
- Alpaca Markets, Market Data API v2, IEX feed (free plan). 1-minute AMD bars, split-adjusted; SPY, QQQ and NVDA 1-minute bars as context (Phase 5). https://docs.alpaca.markets/docs/about-market-data-api

## Libraries
Versions match the project `.venv`.

| Library | Version | Used for |
|---------|---------|----------|
| alpaca-py | 0.34.0 | Downloading historical bars from Alpaca |
| pandas | 2.2.3 | Time-series data handling |
| NumPy | 1.26.4 | Numerical operations |
| PyArrow | 17.0.0 | Parquet storage |
| exchange-calendars | 4.5.4 | NYSE (XNYS) sessions and half-days for the minute grid |
| PyYAML | 6.0.2 | Reading `config.yaml` |
| python-dotenv | 1.0.1 | Loading API keys from `.env` |
| pytz | 2026.6 | Time-zone conversion to America/New_York |
| LightGBM | 4.5.0 | Gradient-boosted trees (Phase 5) |
| scikit-learn | 1.5.2 | Logistic regression, StandardScaler, Pipeline (Phase 4) |
| Matplotlib | 3.9.2 | EDA figures in `report/figures/` |
| pytest | 8.4.2 | Automated tests (leakage, session selection) |

Libraries are added here when they are first used in code.

## Papers
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017).
  LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural
  Information Processing Systems 30* (NeurIPS 2017).
