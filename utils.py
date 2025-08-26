# utils.py
# Modular helper functions extracted from EDA_Cleaned_Notebook

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from textblob import TextBlob
from datetime import datetime, timedelta

def download_market_data(tickers_dict, start_date, end_date):
    """
    Download Close price market data for multiple tickers using yfinance.

    Parameters:
    -----------
    tickers_dict : dict
        Dictionary of {name: ticker_symbol}
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format

    Returns:
    --------
    pd.DataFrame : Close prices with renamed columns
    dict : Failed downloads with error messages
    """
    print("\n📈 Downloading market data...")

    # Validate date format
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be in 'YYYY-MM-DD' format.")

    # Extract tickers and reverse-lookup for name remapping
    ticker_list = list(tickers_dict.values())
    reverse_map = {v: k for k, v in tickers_dict.items()}

    try:
        # Batch download all tickers at once
        data = yf.download(ticker_list, start=start_date, end=end_date, progress=False)

        # Handle single vs multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close']
        else:
            close_prices = data.dropna(how='all')  # fallback if only 1 ticker

        # Drop completely empty columns
        close_prices = close_prices.dropna(axis=1, how='all')

        # Rename columns to original names
        close_prices.rename(columns=reverse_map, inplace=True)

        # Filter out tickers with insufficient data
        failed_downloads = {}
        for name in tickers_dict.keys():
            if name not in close_prices.columns:
                failed_downloads[name] = "No data downloaded"
            elif close_prices[name].dropna().shape[0] < 100:
                failed_downloads[name] = f"Insufficient data ({close_prices[name].dropna().shape[0]} points)"
                close_prices.drop(columns=name, inplace=True)

        if not close_prices.empty:
            print(f"✅ Successfully downloaded {close_prices.shape[1]} out of {len(tickers_dict)} tickers.")
        else:
            print("❌ No valid data found.")

        # Log failures
        if failed_downloads:
            print(f"\n⚠️  Failed downloads: {len(failed_downloads)}")
            for name, reason in failed_downloads.items():
                print(f"   • {name}: {reason}")

        return close_prices if not close_prices.empty else None, failed_downloads

    except Exception as e:
        print(f"\n❌ Error downloading data: {e}")
        return None, {name: "Batch download failed" for name in tickers_dict}


market_data, failed_downloads = download_market_data(TICKERS, START_DATE, END_DATE)

if market_data is None:
  print("❌ Failed to download any market data. Exiting.")

market_data.head()

def clean_and_process_data(market_data, add_sentiment=True):
    """
    Clean market data and add engineered features

    Parameters:
    -----------
    market_data : pd.DataFrame
        Raw market data
    add_sentiment : bool
        Whether to add sentiment features

    Returns:
    --------
    pd.DataFrame : Cleaned and processed data
    """
    print(f"\n🧹 Cleaning and processing data...")
    print(f"   Initial shape: {market_data.shape}")

    # Create copy for processing
    df = market_data.copy()

    # Handle missing values
    print(f"   Missing values before cleaning:")
    missing_before = df.isnull().sum()
    for col, missing in missing_before[missing_before > 0].items():
        print(f"     • {col}: {missing:,} ({missing/len(df)*100:.1f}%)")

    # Forward fill missing values (common for financial data)
    df = df.fillna(method='ffill')

    # Backward fill any remaining NaN at the beginning
    df = df.fillna(method='bfill')

    # Drop any remaining NaN rows
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)

    if rows_before != rows_after:
        print(f"   Removed {rows_before - rows_after:,} rows with NaN values")

    # Add sentiment features if requested
    if add_sentiment:
        print(f"   Adding sentiment features...")

        # Generate sentiment data
        sentiment_series = generate_sentiment_data(df.index)
        df['News_Sentiment'] = sentiment_series

        # Create sentiment-based features
        df['Sentiment_7d_MA'] = df['News_Sentiment'].rolling(window=7, min_periods=1).mean()
        df['Sentiment_30d_MA'] = df['News_Sentiment'].rolling(window=30, min_periods=1).mean()
        df['Sentiment_Volatility'] = df['News_Sentiment'].rolling(window=30, min_periods=1).std()

        print(f"   ✅ Added 4 sentiment-based features")

    print(f"   Final shape: {df.shape}")
    print(f"   Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    return df

def generate_sentiment_data(date_index, seed=42):
    """
    Generate simulated sentiment data for demonstration

    In production, replace this with real news sentiment analysis:
    - News API integration
    - NLP sentiment analysis
    - Social media sentiment
    - Economic reports sentiment

    Parameters:
    -----------
    date_index : pd.DatetimeIndex
        Date index to match market data
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.Series : Sentiment scores indexed by date
    """
    print(f"\n📰 Generating simulated sentiment data...")

    np.random.seed(seed)
    n_days = len(date_index)

    # Create more realistic sentiment with autocorrelation
    sentiment_scores = np.zeros(n_days)
    sentiment_scores[0] = np.random.normal(0, 0.3)

    # AR(1) process for persistence
    for i in range(1, n_days):
        sentiment_scores[i] = (0.7 * sentiment_scores[i-1] +
                              0.3 * np.random.normal(0, 0.3))

    # Add occasional extreme events (simulate major news)
    extreme_events = np.random.choice(n_days, size=int(n_days * 0.01), replace=False)
    sentiment_scores[extreme_events] += np.random.choice([-1.2, 1.2], size=len(extreme_events))

    # Clip to reasonable range
    sentiment_scores = np.clip(sentiment_scores, -2, 2)

    sentiment_series = pd.Series(sentiment_scores, index=date_index, name='News_Sentiment')

    print(f"✅ Generated sentiment data with:")
    print(f"   • Mean: {sentiment_series.mean():.4f}")
    print(f"   • Std:  {sentiment_series.std():.4f}")
    print(f"   • Min:  {sentiment_series.min():.4f}")
    print(f"   • Max:  {sentiment_series.max():.4f}")

    return sentiment_series

def analyze_text_sentiment(text):
    """
    Analyze sentiment using TextBlob (for future real implementation)

    Parameters:
    -----------
    text : str
        Text to analyze

    Returns:
    --------
    float : Sentiment polarity score (-1 to 1)
    """
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except:
        return 0.0

processed_data = clean_and_process_data(market_data, add_sentiment=True)


def analyze_feature_relationships_with_oil(data, target_col='WTI_Crude', threshold=0.3, top_n=10):
    """
    Perform feature correlation and importance analysis with respect to oil prices.

    Parameters:
    -----------
    data : pd.DataFrame
        Market data (features as columns).
    target_col : str
        Name of the target feature (default: 'WTI_Crude').
    threshold : float
        Absolute correlation threshold to select relevant features.
    top_n : int
        Number of top correlated features to display/analyze.

    Returns:
    --------
    selected_features : list
        List of selected feature names (including target).
    correlations : pd.Series
        Series of feature correlations with target_col.
    """
    if target_col not in data.columns:
        print(f"❌ Target column '{target_col}' not found in data.")
        return None, None

    print(f"\n🎯 Analyzing feature relationships with {target_col}...")

    # 1. Correlation Matrix
    corr_matrix = data.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title("📌 Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # 2. Correlation with Target
    correlations = corr_matrix[target_col].drop(target_col)
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print(f"\n🔥 Top {top_n} Correlations with {target_col}:")
    for i, (feature, abs_corr) in enumerate(correlations_abs.head(top_n).items(), 1):
        actual_corr = correlations[feature]
        print(f"   {i:2}. {feature:25}: {actual_corr:+.3f} (|{abs_corr:.3f}|)")

    # 3. Bar Chart of Top-N Correlations
    plt.figure(figsize=(10, 6))
    top_corr = correlations_abs.head(top_n)
    sns.barplot(x=top_corr.values, y=top_corr.index, orient='h', palette='coolwarm')
    plt.title(f"📊 Top {top_n} Absolute Correlations with {target_col}")
    plt.xlabel("Correlation (absolute)")
    plt.tight_layout()
    plt.show()

    # 4. Feature Selection
    selected_features_corr = correlations_abs[correlations_abs >= threshold]
    selected_features = [target_col] + list(selected_features_corr.index)

    print(f"\n✅ Selected {len(selected_features)-1} features with |correlation| ≥ {threshold}")

    # 5. Pairplot for selected features
    if len(selected_features) <= 10:
        sns.pairplot(data[selected_features].dropna())
        plt.suptitle("📷 Pairwise Relationships (Selected Features)", y=1.02)
        plt.show()
    else:
        print("📷 Skipping pairplot (too many features selected).")

    # 6. Feature Importance (Random Forest)
    print("\n🌲 Calculating feature importances (Random Forest)...")
    feature_data = data[selected_features].dropna()
    X = feature_data.drop(columns=target_col)
    y = feature_data[target_col]

    # Normalize features for importance comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_scaled, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='teal')
    plt.title("🌟 Feature Importances (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    # 7. Histogram/Distribution of Selected Features
    print("\n📈 Plotting distributions of selected features...")
    selected_data = data[selected_features].dropna()
    selected_data.hist(figsize=(15, 10), bins=30, layout=(len(selected_features) // 3 + 1, 3))
    plt.suptitle("📊 Distributions of Selected Features", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
        # 8. Scatterplots vs. Target
    print("\n📌 Scatterplots of selected features vs. target...")
    max_plots = min(9, len(selected_features) - 1)
    fig, axes = plt.subplots(nrows=(max_plots + 2) // 3, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(selected_features_corr.head(max_plots).index):
        ax = axes[i]
        sns.scatterplot(data=data, x=feature, y=target_col, ax=ax, alpha=0.5)
        corr_val = correlations[feature]
        ax.set_title(f"{feature} (r = {corr_val:.2f})")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"📉 Scatterplots vs. {target_col}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


    return selected_features, correlations


selected_features, correlations = analyze_feature_relationships_with_oil(
            processed_data, target_col='WTI_Crude', threshold=0.3
        )
