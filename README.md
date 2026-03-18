# GOOG Sentiment Analysis

A machine learning pipeline that analyzes news sentiment about Google (GOOG) stock and correlates it with next-day price returns.

## Overview

This project uses FinBERT, a BERT model fine-tuned on financial text, to classify news articles about GOOG as positive, neutral, or negative. The resulting sentiment scores are then correlated with next-day stock returns to explore whether news sentiment has predictive power.

## Pipeline

1. **Data Collection** — Fetches GOOG price history via `yfinance` and news articles via NewsAPI
2. **Sentiment Analysis** — Runs FinBERT on each article title to classify sentiment
3. **Correlation Analysis** — Aggregates daily sentiment and correlates with next-day returns
4. **Visualization** — Plots price, sentiment over time, and sentiment vs return

## Results

- Pearson correlation between daily sentiment and next-day return: **0.2275**
- Weak positive correlation — consistent with the hypothesis that positive news tends to precede slight price increases
- Limited by 30-day data window (NewsAPI free tier)

## Tech Stack

- `yfinance` — stock price data
- `newsapi-python` — financial news
- `transformers` — FinBERT model (ProsusAI/finbert)
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization

## Setup
```bash
git clone https://github.com/JoseLuiSS0/GOOG-sentiment-analysis.git
cd GOOG-sentiment-analysis
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```
NEWSAPI_KEY=your_api_key_here
```

Then run the notebooks in this order:
```
01_data_collection.ipynb
02_sentiment_analysis.ipynb
03_correlation_analysis.ipynb
04_visualization.ipynb
```

## Limitations

- NewsAPI free tier limits data to the last 30 days (~13 trading days with news coverage)
- Small sample size limits statistical significance
- Sentiment is computed on headlines only, not full article body

## Next Steps

- Scale to 1+ year of data using Alpha Vantage or a paid NewsAPI plan
- Add SHAP values for model interpretability
- Extend to other tickers for comparison
