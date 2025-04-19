import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_financial_news(ticker, start_date, end_date, api_key=None):
    """
    Fetch financial news related to a specific stock using News API
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    api_key : str, optional
        News API key. If not provided, tries to get from environment variable
    
    Returns:
    --------
    pandas.DataFrame
        Financial news articles
    """
    if api_key is None:
        api_key = os.getenv("NEWS_API_KEY")
        if api_key is None:
            print("No API key provided and none found in environment variables")
            return None
    
    # Convert dates to required format (YYYY-MM-DD)
    start_date_fmt = start_date
    end_date_fmt = end_date
    
    # Prepare API request
    url = "https://newsapi.org/v2/everything"
    
    # For companies, we'll search both the ticker and company name
    company_names = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'META': 'Meta'
        # Add more as needed
    }
    
    company_name = company_names.get(ticker, ticker)
    query = f"{ticker} OR {company_name} stock"
    
    params = {
        'q': query,
        'from': start_date_fmt,
        'to': end_date_fmt,
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
        
        if data['totalResults'] == 0:
            print(f"No news found for {ticker} between {start_date} and {end_date}")
            return pd.DataFrame()
        
        # Create DataFrame from the articles
        articles = data['articles']
        df = pd.DataFrame(articles)
        
        # Extract relevant columns and add ticker
        df = df[['source', 'title', 'description', 'url', 'publishedAt']]
        df['source'] = df['source'].apply(lambda x: x.get('name', '') if isinstance(x, dict) else x)
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df['publishedAt']).dt.date
        
        print(f"Retrieved {len(df)} news articles for {ticker}")
        return df
    
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return None

def get_news_for_multiple_tickers(tickers, start_date, end_date, api_key=None):
    """
    Fetch news for multiple stock tickers
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    api_key : str, optional
        News API key
    
    Returns:
    --------
    pandas.DataFrame
        Combined news data
    """
    all_news = []
    
    for ticker in tickers:
        news_data = get_financial_news(ticker, start_date, end_date, api_key)
        if news_data is not None and not news_data.empty:
            all_news.append(news_data)
        # Sleep to respect API rate limits
        time.sleep(1)
    
    if not all_news:
        print("No news was collected for any of the tickers")
        return None
    
    # Combine all DataFrames
    combined_news = pd.concat(all_news)
    combined_news = combined_news.reset_index(drop=True)
    
    return combined_news

if __name__ == "__main__":
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    news_data = get_news_for_multiple_tickers(tickers, start_date, end_date)
    
    if news_data is not None:
        # Save to CSV
        news_data.to_csv('data/raw/news_data.csv', index=False)
        print(f"News data saved to data/raw/news_data.csv")
```

`src/data/preprocess.py`:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_stock_data(stock_df):
    """
    Preprocess stock data for model training
    
    Parameters:
    -----------
    stock_df : pandas.DataFrame
        Raw stock data
    
    Returns:
    --------
    pandas.DataFrame
        Processed stock data
    """
    # Create a copy to avoid modifying the original
    df = stock_df.copy()
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Date and Ticker
    df = df.sort_values(['Ticker', 'Date'])
    
    # Handle missing values
    df = df.dropna()
    
    # Create target variable: price direction for next day (1 if price goes up, 0 if down)
    df['Target'] = df.groupby('Ticker')['Adj Close'].shift(-1)
    df['Target'] = (df['Target'] > df['Adj Close']).astype(int)
    
    # Create a new column for the day of the week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Create a new column for the month
    df['Month'] = df['Date'].dt.month
    
    # Calculate additional technical indicators
    # Bollinger Bands
    df['20d_std'] = df.groupby('Ticker')['Adj Close'].rolling(window=20).std().reset_index(0, drop=True)
    df['Upper_Band'] = df['MA_20'] + (df['20d_std'] * 2)
    df['Lower_Band'] = df['MA_20'] - (df['20d_std'] * 2)
    
    # Price/Volume indicators
    df['Price_Change'] = df['Adj Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Drop rows with NaN values that may have been introduced
    df = df.dropna()
    
    return df

def preprocess_news_text(text):
    """
    Preprocess news text for sentiment analysis
    
    Parameters:
    -----------
    text : str
        News text
    
    Returns:
    --------
    str
        Processed text
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin into a string
    processed_text = ' '.join(tokens)
    
    return processed_text

def preprocess_news_data(news_df):
    """
    Preprocess news data for sentiment analysis
    
    Parameters:
    -----------
    news_df : pandas.DataFrame
        Raw news data
    
    Returns:
    --------
    pandas.DataFrame
        Processed news data
    """
    # Create a copy to avoid modifying the original
    df = news_df.copy()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Process title and description
    df['processed_title'] = df['title'].apply(preprocess_news_text)
    df['processed_description'] = df['description'].apply(preprocess_news_text)
    
    # Combine title and description for analysis
    df['processed_content'] = df['processed_title'] + ' ' + df['processed_description']
    
    return df

def merge_stock_and_news_data(stock_df, news_df):
    """
    Merge stock and news data by date and ticker
    
    Parameters:
    -----------
    stock_df : pandas.DataFrame
        Processed stock data
    news_df : pandas.DataFrame
        Processed news data with sentiment scores
    
    Returns:
    --------
    pandas.DataFrame
        Merged dataset
    """
    # Ensure date formats match
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    news_df['date'] = pd.to_datetime(news_df['date'])
    
    # Group news by date and ticker, and calculate mean sentiment scores
    news_agg = news_df.groupby(['date', 'ticker']).agg({
        'sentiment_score': 'mean',
        'positive_score': 'mean',
        'negative_score': 'mean',
        'neutral_score': 'mean'
    }).reset_index()
    
    # Rename columns for consistency
    news_agg.columns = ['Date', 'Ticker', 'Sentiment_Score', 'Positive_Score', 'Negative_Score', 'Neutral_Score']
    
    # Merge the datasets
    merged_df = pd.merge(stock_df, news_agg, on=['Date', 'Ticker'], how='left')
    
    # Fill NaN sentiment scores with 0
    sentiment_cols = ['Sentiment_Score', 'Positive_Score', 'Negative_Score', 'Neutral_Score']
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)
    
    return merged_df

def scale_features(df, feature_columns, scaler_type='minmax'):
    """
    Scale numerical features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    feature_columns : list
        List of columns to scale
    scaler_type : str, optional
        Type of scaler to use ('minmax' or 'standard')
    
    Returns:
    --------
    pandas.DataFrame, object
        Scaled dataframe and the scaler object
    """
    # Create a copy to avoid modifying the original
    df_scaled = df.copy()
    
    # Select the appropriate scaler
    if scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be either 'minmax' or 'standard'")
    
    # Scale the features
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df_scaled, scaler
