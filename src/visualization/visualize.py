import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.dates as mdates
import os

# Set style for matplotlib
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')

def plot_stock_price_history(df, ticker=None, date_col='Date', price_col='Adj Close', save_path=None):
    """
    Plot stock price history
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data
    ticker : str, optional
        Ticker to filter by. If None, plots all tickers
    date_col : str, optional
        Date column
    price_col : str, optional
        Price column
    save_path : str, optional
        Path to save the figure
    """
    # Filter by ticker if provided
    if ticker:
        plot_df = df[df['Ticker'] == ticker].copy()
        title = f'{ticker} Stock Price History'
    else:
        plot_df = df.copy()
        title = 'Stock Price History'
    
    # Convert date to datetime if not already
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Plot for each ticker
    for tick in plot_df['Ticker'].unique():
        tick_df = plot_df[plot_df['Ticker'] == tick]
        plt.plot(tick_df[date_col], tick_df[price_col], label=tick)
    
    # Formatting
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_price_with_ma(df, ticker, date_col='Date', price_col='Adj Close', ma_cols=['MA_5', 'MA_20'], save_path=None):
    """
    Plot stock price with moving averages
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data
    ticker : str
        Ticker to plot
    date_col : str, optional
        Date column
    price_col : str, optional
        Price column
    ma_cols : list, optional
        Moving average columns
    save_path : str, optional
        Path to save the figure
    """
    # Filter by ticker
    plot_df = df[df['Ticker'] == ticker].copy()
    
    # Convert date to datetime if not already
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Plot price
    plt.plot(plot_df[date_col], plot_df[price_col], label='Price', linewidth=1.5)
    
    # Plot moving averages
    for ma_col in ma_cols:
        plt.plot(plot_df[date_col], plot_df[ma_col], label=ma_col, linewidth=1.5, alpha=0.7)
    
    # Formatting
    plt.title(f'{ticker} Stock Price with Moving Averages', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_technical_indicators(df, ticker, date_col='Date', indicators=None, save_path=None):
    """
    Plot technical indicators
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data
    ticker : str
        Ticker to plot
    date_col : str, optional
        Date column
    indicators : dict, optional
        Dictionary mapping indicator names to columns
    save_path : str, optional
        Path to save the figure
    """
    # Filter by ticker
    plot_df = df[df['Ticker'] == ticker].copy()
    
    # Convert date to datetime if not already
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    
    # Default indicators if none provided
    if indicators is None:
        indicators = {
            'RSI': 'RSI',
            'MACD': 'MACD',
            'MACD Signal': 'MACD_Signal'
        }
    
    # Create subplots
    fig, axes = plt.subplots(len(indicators), 1, figsize=(14, 4 * len(indicators)), sharex=True)
    
    # If there's only one indicator, wrap axes in a list
    if len(indicators) == 1:
        axes = [axes]
    
    # Plot each indicator
    for i, (name, col) in enumerate(indicators.items()):
        axes[i].plot(plot_df[date_col], plot_df[col], label=name)
        axes[i].set_title(name, fontsize=14)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    # Add horizontal line at 30 and 70 for RSI
    if 'RSI' in indicators.values():
        rsi_idx = list(indicators.values()).index('RSI')
        axes[rsi_idx].axhline(y=30, color='g', linestyle='--', alpha=0.5)
        axes[rsi_idx].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    
    # Add horizontal line at 0 for MACD
    if 'MACD' in indicators.values():
        macd_idx = list(indicators.values()).index('MACD')
        axes[macd_idx].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Format x-axis dates
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.xlabel('Date', fontsize=12)
    plt.suptitle(f'{ticker} Technical Indicators', fontsize=16)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_correlation_matrix(df, columns=None, title='Feature Correlation Matrix', save_path=None):
    """
    Plot correlation matrix
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data
    columns : list, optional
        Columns to include in correlation matrix
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    # Select columns if provided
    if columns:
        corr_df = df[columns].corr()
    else:
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_df = df[numeric_cols].corr()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_df,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8}
    )
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_sentiment_over_time(df, ticker=None, date_col='date', sentiment_col='sentiment_score', save_path=None):
    """
    Plot sentiment scores over time
    
    Parameters:
    -----------
    df : pandas.DataFrame
        News data with sentiment scores
    ticker : str, optional
        Ticker to filter by. If None, plots all tickers
    date_col : str, optional
        Date column
    sentiment_col : str, optional
        Sentiment column
    save_path : str, optional
        Path to save the figure
    """
    # Filter by ticker if provided
    if ticker:
        plot_df = df[df['ticker'] == ticker].copy()
        title = f'{ticker} Sentiment Over Time'
    else:
        plot_df = df.copy()
        title = 'Sentiment Over Time by Ticker'
    
    # Convert date to datetime if not already
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    
    # Group by date and ticker, calculate mean sentiment
    grouped = plot_df.groupby([date_col, 'ticker'])[sentiment_col].mean().reset_index()
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Plot for each ticker
    for tick in grouped['ticker'].unique():
        tick_df = grouped[grouped['ticker'] == tick]
        plt.plot(tick_df[date_col], tick_df[sentiment_col], marker='o', linestyle='-', label=tick)
    
    # Add horizontal line at 0
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Formatting
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sentiment Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_price_vs_sentiment(df, ticker, date_col='Date', price_col='Adj Close', sentiment_col='Sentiment_Score', save_path=None):
    """
    Plot stock price vs sentiment
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Merged stock and news data
    ticker : str
        Ticker to plot
    date_col : str, optional
        Date column
    price_col : str, optional
        Price column
    sentiment_col : str, optional
        Sentiment column
    save_path : str, optional
        Path to save the figure
    """
    # Filter by ticker
    plot_df = df[df['Ticker'] == ticker].copy()
    
    # Convert date to datetime if not already
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot price on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', color=color, fontsize=12)
    ax1.plot(plot_df[date_col], plot_df[price_col], color=color, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create secondary y-axis for sentiment
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sentiment Score', color=color, fontsize=12)
    ax2.plot(plot_df[date_col], plot_df[sentiment_col], color=color, label='Sentiment', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add horizontal line at 0 for sentiment
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Formatting
    fig.suptitle(f'{ticker} Stock Price vs News Sentiment', fontsize=16)
    fig.tight_layout()
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_model_comparison(model_metrics, metric='accuracy', save_path=None):
    """
    Plot model comparison
    
    Parameters:
    -----------
    model_metrics : dict
        Dictionary with model names as keys and metric dictionaries as values
    metric : str, optional
        Metric to plot
    save_path : str, optional
        Path to save the figure
    """
    # Extract model names and metric values
    models = list(model_metrics.keys())
    values = [metrics[metric] for metrics in model_metrics.values()]
    
    # Sort by metric value
    sorted_indices = np.argsort(values)[::-1]  # Descending order
    models = [models[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    bars = plt.bar(models, values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Formatting
    plt.title(f'Model Comparison by {metric.capitalize()}', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.ylim(0, max(values) * 1.1)  # Add some space for the labels
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, title='Feature Importance', save_path=None):
    """
    Plot feature importance
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, optional
        Number of top features to show
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    # Extract feature importances
    importances = model.feature_importances_
    
    # Create dataframe for easier sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Select top N features
    importance_df = importance_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot
    sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df,
        palette='viridis'
    )
    
    # Formatting
    plt.title(title, fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=None, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    classes : list, optional
        List of class names
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Default class names if none provided
    if classes is None:
        classes = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    
    # Formatting
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    save_path : str, optional
        Path to save the figure
    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot precision-recall curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    save_path : str, optional
        Path to save the figure
    """
    # Compute precision-recall curve and area
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot precision-recall curve
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_interactive_price_plot(df, ticker, date_col='Date', price_col='Adj Close', ma_cols=None, save_path=None):
    """
    Create interactive price plot with Plotly
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data
    ticker : str
        Ticker to plot
    date_col : str, optional
        Date column
    price_col : str, optional
        Price column
    ma_cols : list, optional
        Moving average columns
    save_path : str, optional
        Path to save the figure as HTML
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Filter by ticker
    plot_df = df[df['Ticker'] == ticker].copy()
    
    # Convert date to datetime if not already
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    
    # Default moving average columns if none provided
    if ma_cols is None:
        ma_cols = ['MA_5', 'MA_20']
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=plot_df[date_col],
            y=plot_df[price_col],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add moving average lines
    for ma_col in ma_cols:
        if ma_col in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df[date_col],
                    y=plot_df[ma_col],
                    mode='lines',
                    name=ma_col,
                    line=dict(width=1.5, dash='dash')
                )
            )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price History',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=3, label='3m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')
            ])
        )
    )
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
    
    return fig
