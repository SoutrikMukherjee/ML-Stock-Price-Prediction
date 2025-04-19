import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

def create_traditional_ml_models():
    """
    Create traditional ML models
    
    Returns:
    --------
    dict
        Dictionary of initialized models
    """
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    return models

def create_lstm_model(input_shape, num_classes=1):
    """
    Create LSTM model
    
    Parameters:
    -----------
    input_shape : tuple
        Input shape (timesteps, features)
    num_classes : int, optional
        Number of classes (1 for binary)
    
    Returns:
    --------
    tensorflow.keras.models.Sequential
        LSTM model
    """
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    
    # Output layer
    if num_classes == 1:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def create_gru_model(input_shape, num_classes=1):
    """
    Create GRU model
    
    Parameters:
    -----------
    input_shape : tuple
        Input shape (timesteps, features)
    num_classes : int, optional
        Number of classes (1 for binary)
    
    Returns:
    --------
    tensorflow.keras.models.Sequential
        GRU model
    """
    model = Sequential()
    
    # GRU layers
    model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dropout(0.2))
    
    # Output layer
    if num_classes == 1:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def prepare_sequence_data(X, y, sequence_length=10):
    """
    Prepare sequence data for LSTM/GRU
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Target
    sequence_length : int, optional
        Sequence length
    
    Returns:
    --------
    numpy.ndarray, numpy.ndarray
        Sequence features and targets
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

def train_traditional_ml_models(X_train, y_train, models=None, param_grid=None):
    """
    Train traditional ML models
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training targets
    models : dict, optional
        Dictionary of initialized models
    param_grid : dict, optional
        Dictionary of parameter grids for each model
    
    Returns:
    --------
    dict
        Dictionary of trained models
    """
    # Default models if none provided
    if models is None:
        models = create_traditional_ml_models()
    
    # Default parameter grids if none provided
    if param_grid is None:
        param_grid = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
            'mlp': {
                'hidden_layer_sizes': [(50, 25), (100, 50)],
                'alpha': [0.0001, 0.001]
            }
        }
    
    # Train models
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use GridSearchCV with TimeSeriesSplit for hyperparameter tuning
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(
            model,
            param_grid[name],
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        trained_models[name] = best_model
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return trained_models

def train_deep_learning_models(X_train, y_train, sequence_length=10, models_to_train=None, batch_size=32, epochs=50, validation_split=0.2, save_dir=None):
    """
    Train deep learning models
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training targets
    sequence_length : int, optional
        Sequence length for LSTM/GRU
    models_to_train : list, optional
        List of model types to train ('lstm', 'gru', 'bidirectional_lstm')
    batch_size : int, optional
        Batch size
    epochs : int, optional
        Number of epochs
    validation_split : float, optional
        Validation split
    save_dir : str, optional
        Directory to save models
    
    Returns:
    --------
    dict
        Dictionary of trained models and their histories
    """
    # Default models to train if none provided
    if models_to_train is None:
        models_to_train = ['lstm', 'gru', 'bidirectional_lstm']
    
    # Prepare sequence data
    X_seq, y_seq = prepare_sequence_data(X_train, y_train, sequence_length)
    
    # Create directory for saved models if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Train models
    trained_models = {}
    
    for model_type in models_to_train:
        print(f"\nTraining {model_type}...")
        
        # Create model
        if model_type == 'lstm':
            model = create_lstm_model((sequence_length, X_train.shape[1]))
        elif model_type == 'gru':
            model = create_gru_model((sequence_length, X_train.shape[1]))
        elif model_type == 'bidirectional_lstm':
            model = Sequential()
            model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(sequence_length, X_train.shape[1])))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(32)))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        else:
            print(f"Unknown model type: {model_type}")
            continue
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        if save_dir:
            model_path = os.path.join(save_dir, f"{model_type}.h5")
            callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
        
        # Train model
        history = model.fit(
            X_seq, y_seq,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        trained_models[model_type] = {
            'model': model,
            'history': history.history
        }
    
    return trained_models

def evaluate_models(models, X_test, y_test, sequence_length=None):
    """
    Evaluate trained models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test targets
    sequence_length : int, optional
        Sequence length for deep learning models
    
    Returns:
    --------
    dict
        Dictionary of model metrics
    """
    metrics = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Deep learning models
        if isinstance(model, dict) and 'model' in model:
            if sequence_length is None:
                raise ValueError("sequence_length must be provided for deep learning models")
            
            # Prepare sequence data
            X_seq, y_seq = prepare_sequence_data(X_test, y_test, sequence_length)
            
            # Get predictions
            y_pred_proba = model['model'].predict(X_seq).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_seq, y_pred)
            precision = precision_score(y_seq, y_pred)
            recall = recall_score(y_seq, y_pred)
            f1 = f1_score(y_seq, y_pred)
            roc_auc = roc_auc_score(y_seq, y_pred_proba)
            
            metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': confusion_matrix(y_seq, y_pred)
            }
            
            # Print metrics
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
        
        # Traditional ML models
        else:
            # Get predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Print metrics
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
    
    return metrics

def save_models(models, save_dir):
    """
    Save trained models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    save_dir : str
        Directory to save models
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save models
    for name, model in models.items():
        # Deep learning models
        if isinstance(model, dict) and 'model' in model:
            model_path = os.path.join(save_dir, f"{name}.h5")
            model['model'].save(model_path)
            
            # Save history separately
            history_path = os.path.join(save_dir, f"{name}_history.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump(model['history'], f)
        
        # Traditional ML models
        else:
            model_path = os.path.join(save_dir, f"{name}.pkl")
            joblib.dump(model, model_path)
