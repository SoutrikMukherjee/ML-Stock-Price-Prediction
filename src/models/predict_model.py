import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime, timedelta

def load_ml_model(model_path):
    """
    Load ML model from path
    
    Parameters:
    -----------
    model_path : str
        Path to the model file
    
    Returns:
    --------
    object
        Loaded model
    """
    return joblib.load(model_path)

def load_dl_model(model_path):
    """
    Load deep learning model from path
    
    Parameters:
    -----------
    model_path : str
        Path to the model file
    
    Returns:
    --------
    tensorflow.keras.models.Model
        Loaded model
    """
    return load_model(model_path)

def prepare_data_for_prediction(data, sequence_length=None, scaler=None, feature_columns=None):
    """
    Prepare data for prediction
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    sequence_length : int, optional
        Sequence length for deep learning models
    scaler : object, optional
        Fitted scaler for feature scaling
    feature_columns : list, optional
        List of feature columns
    
    Returns:
    --------
    numpy.ndarray or tuple
        Prepared data
    """
    # Select features
    if feature_columns:
        X = data[feature_columns].values
    else:
        # Assuming all columns except the target are features
        X = data.select_dtypes(include=[np.number]).values
    
    # Scale features
    if scaler:
        X = scaler.transform(X)
    
    # Prepare sequence data for deep learning
    if sequence_length:
        X_seq = []
        for i in range(len(X) - sequence_length + 1):
            X_seq.append(X[i:i + sequence_length])
        return np.array(X_seq)
    
    return X

def predict_next_day_movement(model, latest_data, is_dl_model=False, sequence_length=None, scaler=None, feature_columns=None):
    """
    Predict next day's price movement
    
    Parameters:
    -----------
    model : object
        Trained model
    latest_data : pandas.DataFrame
        Latest data for prediction
    is_dl_model : bool, optional
        Whether the model is a deep learning model
    sequence_length : int, optional
        Sequence length for deep learning models
    scaler : object, optional
        Fitted scaler for feature scaling
    feature_columns : list, optional
        List of feature columns
    
    Returns:
    --------
    dict
        Prediction results
    """
    # Prepare data
    if is_dl_model:
        if sequence_length is None:
            raise ValueError("sequence_length must be provided for deep learning models")
        
        X = prepare_data_for_prediction(latest_data, sequence_length, scaler, feature_columns)
        
        # Get most recent sequence
        X_latest = X[-1:] if len(X.shape) == 3 else X.reshape(1, sequence_length, -1)
        
        # Make prediction
        prob = model.predict(X_latest)[0][0]
        prediction = 1 if prob > 0.5 else 0
    else:
        X = prepare_data_for_prediction(latest_data, None, scaler, feature_columns)
        
        # Get most recent data point
        X_latest = X[-1].reshape(1, -1)
        
        # Make prediction
        prob = model.predict_proba(X_latest)[0][1]
        prediction = model.predict(X_latest)[0]
    
    # Format results
    result = {
        'prediction': int(prediction),
        'probability': float(prob),
        'direction': 'Up' if prediction == 1 else 'Down',
        'confidence': float(prob if prediction == 1 else 1 - prob),
        'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    }
    
    return result

def ensemble_predictions(predictions, weights=None):
    """
    Create ensemble prediction from multiple models
    
    Parameters:
    -----------
    predictions : list
        List of prediction dictionaries
    weights : list, optional
        List of weights for each model
    
    Returns:
    --------
    dict
        Ensemble prediction
    """
    if weights is None:
        weights = [1.0] * len(predictions)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Calculate weighted average probability
    probabilities = [pred['probability'] for pred in predictions]
    weighted_prob = sum(p * w for p, w in zip(probabilities, weights))
    
    # Make ensemble prediction
    prediction = 1 if weighted_prob > 0.5 else 0
    
    # Format results
    result = {
        'prediction': prediction,
        'probability': weighted_prob,
        'direction': 'Up' if prediction == 1 else 'Down',
        'confidence': weighted_prob if prediction == 1 else 1 - weighted_prob,
        'date': predictions[0]['date'],
        'individual_predictions': predictions
    }
    
    return result

def predict_multiple_days(model, latest_data, days=5, is_dl_model=False, sequence_length=None, scaler=None, feature_columns=None):
    """
    Predict price movement for multiple future days
    
    Parameters:
    -----------
    model : object
        Trained model
    latest_data : pandas.DataFrame
        Latest data for prediction
    days : int, optional
        Number of future days to predict
    is_dl_model : bool, optional
        Whether the model is a deep learning model
    sequence_length : int, optional
        Sequence length for deep learning models
    scaler : object, optional
        Fitted scaler for feature scaling
    feature_columns : list, optional
        List of feature columns
    
    Returns:
    --------
    list
        List of prediction results
    """
    # Copy data to avoid modifying the original
    data = latest_data.copy()
    
    # List to store predictions
    predictions = []
    
    for i in range(days):
        # Predict next day
        result = predict_next_day_movement(
            model,
            data,
            is_dl_model,
            sequence_length,
            scaler,
            feature_columns
        )
        
        # Store prediction
        result['day'] = i + 1
        result['date'] = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
        predictions.append(result)
        
        # Simulate next day data based on prediction
        # This is a simplified approach, as we don't know actual values
        # but we can use the prediction to simulate next day's data
        last_row = data.iloc[-1].copy()
        
        # Update date
        last_row['Date'] = pd.to_datetime(last_row['Date']) + timedelta(days=1)
        
        # Update prices based on prediction (simplified)
        if result['prediction'] == 1:
            # Price up by 0.5-1.5%
            change = np.random.uniform(0.005, 0.015)
            last_row['Open'] *= (1 + change/2)
            last_row['High'] *= (1 + change)
            last_row['Low'] *= (1 + change/3)
            last_row['Close'] *= (1 + change)
            last_row['Adj Close'] *= (1 + change)
        else:
            # Price down by 0.5-1.5%
            change = np.random.uniform(0.005, 0.015)
            last_row['Open'] *= (1 - change/2)
            last_row['High'] *= (1 - change/3)
            last_row['Low'] *= (1 - change)
            last_row['Close'] *= (1 - change)
            last_row['Adj Close'] *= (1 - change)
        
        # Generate some random values for other features
        last_row['Volume'] = last_row['Volume'] * np.random.uniform(0.8, 1.2)
        
        # Update technical indicators (simplified)
        # In a real scenario, these would be calculated properly
        for col in data.columns:
            if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker', 'Return', 'Target']:
                if col.startswith('MA_'):
                    # Moving averages
                    window = int(col.split('_')[1])
                    if window > 1:
                        last_n = data['Adj Close'].iloc[-window+1:].values
                        last_row[col] = (np.sum(last_n) + last_row['Adj Close']) / window
                elif col in ['RSI', 'MACD', 'MACD_Signal']:
                    # Keep the same for simplicity
                    pass
                elif 'Sentiment' in col:
                    # Random sentiment between -0.2 and 0.2
                    last_row[col] = np.random.uniform(-0.2, 0.2)
        
        # Append to data
        data = pd.concat([data, pd.DataFrame([last_row])], ignore_index=True)
    
    return predictions
