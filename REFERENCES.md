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
- Nickell, S. (1981). Biases in Dynamic Models with Fixed Effects. *Econometrica*, 49(6),
  1417-1426. (Why within-day correlations of price level with later returns are biased.)
- Roll, R. (1984). A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient
  Market. *The Journal of Finance*, 39(4), 1127-1139. (Bid-ask bounce and 1-bar reversal.)
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
  arXiv:1908.10063. (Headline sentiment; model `ProsusAI/finbert`.)

## News data and libraries (Phase 5c)
- Alpaca News API (articles from Benzinga): https://docs.alpaca.markets/us/docs/historical-news-data
- PyTorch and Hugging Face Transformers, used only in `src/score_news.py`.

## Literature review for the final research round (2026-11-02)
Details checked against journal or working-paper listings and abstracts; full texts not all read.
- Aït-Sahalia, Y., Fan, J., Xue, L., & Zhou, Y. (2022). How and When are High-Frequency Stock
  Returns Predictable? NBER Working Paper 30366. (A later *Management Science* version exists;
  check its author list before citing it.)
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection
  Bias, Backtest Overfitting and Non-Normality. *The Journal of Portfolio Management*, 40(5), 94-107.
- Bogousslavsky, V. (2016). Infrequent Rebalancing, Return Autocorrelation, and Seasonality.
  *The Journal of Finance*, 71(6), 2967-3006.
- Chinco, A., Clark-Joseph, A. D., & Ye, M. (2019). Sparse Signals in the Cross-Section of Returns.
  *The Journal of Finance*, 74(1), 449-492.
- Chordia, T., Roll, R., & Subrahmanyam, A. (2008). Liquidity and Market Efficiency.
  *Journal of Financial Economics*, 87(2), 249-268.
- Gao, L., Han, Y., Li, S. Z., & Zhou, G. (2018). Market Intraday Momentum.
  *Journal of Financial Economics*, 129(2), 394-414.
- Heston, S. L., Korajczyk, R. A., & Sadka, R. (2010). Intraday Patterns in the Cross-Section of
  Stock Returns. *The Journal of Finance*, 65(4), 1369-1407.

## Move-size forecasting (Phase 5f)
Details from memory and journal listings; check page numbers against the originals before citing.
- Andersen, T. G., & Bollerslev, T. (1997). Intraday Periodicity and Volatility Persistence in
  Financial Markets. *Journal of Empirical Finance*, 4(2-3), 115-158.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of
  Econometrics*, 31(3), 307-327.
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of
  Financial Econometrics*, 7(2), 174-196.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business &
  Economic Statistics*, 13(3), 253-263.
- Duan, N. (1983). Smearing Estimate: A Nonparametric Retransformation Method. *Journal of the
  American Statistical Association*, 78(383), 605-610.
- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance
  of United Kingdom Inflation. *Econometrica*, 50(4), 987-1007.
- Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation.
  *Journal of the American Statistical Association*, 102(477), 359-378.
- Patton, A. J. (2011). Volatility Forecast Comparison Using Imperfect Volatility Proxies.
  *Journal of Econometrics*, 160(1), 246-256.
- Zhang, C., Zhang, Y., Cucuringu, M., & Qian, Z. (2024). Volatility Forecasting with Machine
  Learning and Intraday Commonality. *Journal of Financial Econometrics*, 22(2), 492-530.

## Live app (Phase 7)
- FastAPI 0.115.4 (S. Ramírez, MIT licence), https://fastapi.tiangolo.com
- Uvicorn 0.32.1 (Encode, BSD-3 licence), https://www.uvicorn.org
- SQLite (public domain), via Python's `sqlite3` module, https://sqlite.org
- TradingView Lightweight Charts 4.2.0 (Apache-2.0 licence; its attribution logo is kept on the
  chart), https://github.com/tradingview/lightweight-charts
- Alpaca Market Data API, free plan: IEX 1-minute bars and news (Benzinga), https://docs.alpaca.markets
