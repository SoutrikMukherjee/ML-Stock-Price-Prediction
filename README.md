# Stock Price Prediction with Sentiment Analysis

## Project Overview
This project demonstrates an end-to-end machine learning pipeline that predicts stock price movements by combining technical analysis with sentiment analysis from financial news headlines. It showcases proficiency in Python and the major ML/AI libraries.

## Project Structure
```
stock_prediction/
│
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed datasets
│
├── notebooks/
│   ├── 1_data_collection.ipynb   # Data acquisition scripts
│   ├── 2_data_exploration.ipynb  # EDA with visualizations
│   └── 3_model_development.ipynb # Model building and evaluation
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collect_stock_data.py # Scripts to collect stock data via APIs
│   │   ├── collect_news_data.py  # Scripts to collect financial news
│   │   └── preprocess.py         # Data preprocessing functions
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py     # Feature engineering
│   │   └── sentiment_analysis.py # NLP for sentiment extraction
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py        # Train ML models
│   │   ├── predict_model.py      # Make predictions
│   │   └── evaluate_model.py     # Model evaluation metrics
│   │
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py          # Create visualizations
│
├── models/                       # Saved model files
│   ├── sklearn_models/
│   └── deep_learning_models/
│
├── reports/
│   ├── figures/                  # Generated graphics and figures
│   └── model_performance.md      # Model performance comparison
│
├── requirements.txt              # Dependencies
├── setup.py                      # Make project pip installable
└── README.md                     # Project documentation

## README.md

markdown
# Stock Price Prediction with Sentiment Analysis

This project implements an end-to-end machine learning pipeline for predicting stock price movements by combining technical analysis with sentiment analysis from financial news headlines.

## Project Overview

The stock price prediction system:

1. Collects Data: Fetches historical stock prices from Yahoo Finance and financial news headlines from News API
2. Processes Data: Cleans, transforms, and merges stock and news data
3. Extracts Sentiment: Applies NLP techniques to analyze sentiment from financial news
4. Creates Visualizations**: Generates insightful visualizations of stock trends and sentiment correlations
5. Builds ML Models: Trains and evaluates multiple machine learning models (traditional and deep learning)
6. Makes Predictions: Forecasts future stock price movements with confidence scores

## Key Features

- Comprehensive Data Collection: Fetches both market data and news sentiment
- Advanced Technical Indicators: Calculates RSI, MACD, Bollinger Bands, and more
- Sentiment Analysis: Applies multiple NLP techniques (TextBlob, VADER, optional FinBERT integration)
- Feature Engineering: Creates time-based, lagged, rolling, and derived features
- Model Comparison: Evaluates multiple ML approaches (Logistic Regression, Random Forest, Gradient Boosting, SVM, MLP, LSTM, GRU)
- Ensemble Prediction: Combines multiple models for improved prediction accuracy
- Interactive Visualizations: Includes both static and interactive charts

## Directory Structure

The project follows a modular and organized structure:


stock_prediction/
│
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed datasets
│
├── notebooks/
│   ├── 1_data_collection.ipynb   # Data acquisition scripts
│   ├── 2_data_exploration.ipynb  # EDA with visualizations
│   └── 3_model_development.ipynb # Model building and evaluation
│
├── src/
│   ├── __init__.py
│   ├── data/                     # Data handling modules
│   ├── features/                 # Feature engineering modules
│   ├── models/                   # Model training and prediction modules
│   └── visualization/            # Visualization modules
│
├── models/                       # Saved model files
│
├── reports/                      # Reports and visualizations
│
├── requirements.txt              # Dependencies
├── setup.py                      # Make project pip installable
└── README.md                     # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock_prediction.git
   cd stock_prediction
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package and dependencies:
   ```
   pip install -e .
   ```

4. Set up API keys:
   
   Create a `.env` file in the project root with your API keys:
   ```
   NEWS_API_KEY=your_news_api_key
   ```

## Usage

### Running the Notebooks

The notebooks are numbered in the recommended execution order:

1. **Data Collection** (`notebooks/1_data_collection.ipynb`):
   - Collects stock price data and financial news
   - Performs initial data preprocessing
   - Saves raw and processed data

2. **Data Exploration** (`notebooks/2_data_exploration.ipynb`):
   - Visualizes stock price trends and technical indicators
   - Analyzes sentiment patterns
   - Examines correlations between features

3. **Model Development** (`notebooks/3_model_development.ipynb`):
   - Implements feature engineering
   - Trains multiple prediction models
   - Evaluates and compares model performance
   - Makes predictions for future price movements
