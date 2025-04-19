from setuptools import find_packages, setup
setup(
    name='stock_prediction',
    packages=find_packages(),
    version='0.1.0',
    description='Stock Price Prediction with Sentiment Analysis',
    author='ML Engineer',
    license='MIT',
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'tensorflow>=2.8.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'yfinance>=0.1.70',
        'requests>=2.25.0',
        'python-dotenv>=0.19.0',
        'nltk>=3.6.0',
        'textblob>=0.15.3',
        'transformers>=4.17.0',
        'torch>=1.10.0',
        'plotly>=5.7.0',
        'joblib>=1.1.0'
    ],
    python_requires='>=3.8',
)
