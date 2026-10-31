# References

## Data
- Alpaca Markets, Market Data API v2, IEX feed (free plan). 1-minute AMD bars, split-adjusted. https://docs.alpaca.markets/docs/about-market-data-api

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
| Matplotlib | 3.9.2 | EDA figures in `report/figures/` |
| pytest | 8.4.2 | Automated tests (leakage, session selection) |

Libraries are added here when they are first used in code.

## Papers
_None cited yet._
