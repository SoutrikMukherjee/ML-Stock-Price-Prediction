import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Download necessary NLTK resources
nltk.download('vader_lexicon')

def analyze_sentiment_textblob(text):
    """
    Analyze sentiment using TextBlob
    
    Parameters:
    -----------
    text : str
        Text to analyze
    
    Returns:
    --------
    dict
        Dictionary with sentiment scores
    """
    if pd.isna(text) or not text:
        return {
            'polarity': 0,
            'subjectivity': 0
        }
    
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER
    
    Parameters:
    -----------
    text : str
        Text to analyze
    
    Returns:
    --------
    dict
        Dictionary with sentiment scores
    """
    if pd.isna(text) or not text:
        return {
            'compound': 0,
            'pos': 0,
            'neu': 0,
            'neg': 0
        }
    
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

def analyze_sentiment_finbert(texts, batch_size=8):
    """
    Analyze sentiment using FinBERT
    
    Parameters:
    -----------
    texts : list
        List of texts to analyze
    batch_size : int, optional
        Batch size for processing
    
    Returns:
    --------
    list
        List of dictionaries with sentiment scores
    """
    # Handle empty input
    if not texts:
        return []
    
    # Replace empty texts with a placeholder
    texts = [text if text and not pd.isna(text) else "neutral" for text in texts]
    
    try:
        # Load FinBERT model and tokenizer
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create sentiment analysis pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        
        # Process texts in batches
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = sentiment_pipeline(batch_texts)
            results.extend(batch_results)
        
        # Convert pipeline output to standardized format
        sentiment_scores = []
        for result in results:
            label = result['label']
            score = result['score']
            
            if label == 'positive':
                sentiment_scores.append({
                    'sentiment_score': score,
                    'positive_score': score,
                    'negative_score': 0,
                    'neutral_score': 0
                })
            elif label == 'negative':
                sentiment_scores.append({
                    'sentiment_score': -score,
                    'positive_score': 0,
                    'negative_score': score,
                    'neutral_score': 0
                })
            else:  # neutral
                sentiment_scores.append({
                    'sentiment_score': 0,
                    'positive_score': 0,
                    'negative_score': 0,
                    'neutral_score': score
                })
        
        return sentiment_scores
    
    except Exception as e:
        print(f"Error in FinBERT sentiment analysis: {e}")
        # Fallback to VADER for all texts
        return [analyze_sentiment_vader(text) for text in texts]

def add_sentiment_features(news_df, method='combined'):
    """
    Add sentiment features to news data
    
    Parameters:
    -----------
    news_df : pandas.DataFrame
        News data
    method : str, optional
        Sentiment analysis method ('textblob', 'vader', 'finbert', or 'combined')
    
    Returns:
    --------
    pandas.DataFrame
        News data with sentiment features
    """
    # Create a copy to avoid modifying the original
    df = news_df.copy()
    
    if method == 'textblob':
        # Apply TextBlob sentiment analysis
        sentiments = df['processed_content'].apply(analyze_sentiment_textblob)
        df['sentiment_score'] = sentiments.apply(lambda x: x['polarity'])
        df['subjectivity'] = sentiments.apply(lambda x: x['subjectivity'])
        
        # Create positive/negative/neutral scores based on polarity
        df['positive_score'] = df['sentiment_score'].apply(lambda x: max(0, x))
        df['negative_score'] = df['sentiment_score'].apply(lambda x: max(0, -x))
        df['neutral_score'] = df['sentiment_score'].apply(lambda x: 1 - abs(x))
        
    elif method == 'vader':
        # Apply VADER sentiment analysis
        sentiments = df['processed_content'].apply(analyze_sentiment_vader)
        df['sentiment_score'] = sentiments.apply(lambda x: x['compound'])
        df['positive_score'] = sentiments.apply(lambda x: x['pos'])
        df['negative_score'] = sentiments.apply(lambda x: x['neg'])
        df['neutral_score'] = sentiments.apply(lambda x: x['neu'])
        
    elif method == 'finbert':
        # Apply FinBERT sentiment analysis
        texts = df['processed_content'].tolist()
        sentiments = analyze_sentiment_finbert(texts)
        
        df['sentiment_score'] = [s['sentiment_score'] for s in sentiments]
        df['positive_score'] = [s['positive_score'] for s in sentiments]
        df['negative_score'] = [s['negative_score'] for s in sentiments]
        df['neutral_score'] = [s['neutral_score'] for s in sentiments]
        
    elif method == 'combined':
        # Apply both TextBlob and VADER, then average the scores
        textblob_sentiments = df['processed_content'].apply(analyze_sentiment_textblob)
        vader_sentiments = df['processed_content'].apply(analyze_sentiment_vader)
        
        df['textblob_score'] = textblob_sentiments.apply(lambda x: x['polarity'])
        df['vader_score'] = vader_sentiments.apply(lambda x: x['compound'])
        
        # Average the sentiment scores
        df['sentiment_score'] = (df['textblob_score'] + df['vader_score']) / 2
        
        # For positive/negative/neutral, use VADER scores as they're more specialized
        df['positive_score'] = vader_sentiments.apply(lambda x: x['pos'])
        df['negative_score'] = vader_sentiments.apply(lambda x: x['neg'])
        df['neutral_score'] = vader_sentiments.apply(lambda x: x['neu'])
        
    else:
        raise ValueError("method must be one of 'textblob', 'vader', 'finbert', or 'combined'")
    
    return df
