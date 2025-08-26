"""
Oil Price Prediction Utility Functions
=====================================

This module contains all the utility functions needed for oil price prediction,
including data preprocessing, feature engineering, backtesting, and forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import logging

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Time Series Libraries
from prophet import Prophet
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Suppress verbose informational messages from Prophet and its backend
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


class DataProcessor:
    """Handles data preprocessing and resampling operations."""
    
    @staticmethod
    def preprocess_data(df, resample_frequency='M', date_col=None):
        """
        Preprocesses the dataset by setting date column and resampling.
        
        Args:
            df (pd.DataFrame): Input dataframe
            resample_frequency (str): 'D', 'W', or 'M' for daily, weekly, monthly
            date_col (str): Date column name, auto-detected if None
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df = df.copy()
        
        # Auto-detect date column
        if date_col is None:
            date_col = 'Date' if 'Date' in df.columns else 'date'
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        
        # Resample the data to the desired frequency
        df = df.resample(resample_frequency).mean()
        
        # Forward-fill any missing values
        df = df.ffill().reset_index()
        
        return df
    
    @staticmethod
    def get_frequency_config(resample_frequency):
        """
        Returns configuration parameters based on frequency.
        
        Args:
            resample_frequency (str): 'D', 'W', or 'M'
            
        Returns:
            dict: Configuration dictionary
        """
        configs = {
            'D': {
                'freq_str': 'Day',
                'lags': [1, 2, 3, 5, 7, 14],
                'window_sizes': [5, 10, 20],
                'initial_train_periods': 500,
                'test_periods': 200,
                'forecast_periods': 90,
                'seasonal_period': 7
            },
            'W': {
                'freq_str': 'Week',
                'lags': [1, 2, 3, 4],
                'window_sizes': [4, 8, 12],
                'initial_train_periods': 104,
                'test_periods': 52,
                'forecast_periods': 12,
                'seasonal_period': 52
            },
            'M': {
                'freq_str': 'Month',
                'lags': [1, 2, 3, 6],
                'window_sizes': [3, 6, 12],
                'initial_train_periods': 36,
                'test_periods': 24,
                'forecast_periods': 6,
                'seasonal_period': 12
            }
        }
        
        if resample_frequency not in configs:
            raise ValueError("resample_frequency must be 'D', 'W', or 'M'")
            
        return configs[resample_frequency]


class FeatureEngineer:
    """Handles feature engineering operations."""
    
    @staticmethod
    def create_technical_features(data, target_col='WTI_Crude', window_sizes=[5, 10, 20]):
        """
        Creates technical features using only past data to prevent leakage.
        
        Args:
            data (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            window_sizes (list): Window sizes for rolling calculations
            
        Returns:
            pd.DataFrame: Dataframe with technical features
        """
        data = data.copy()
        for window in window_sizes:
            # Shift by 1 to prevent data leakage
            data[f'{target_col}_MA_{window}'] = data[target_col].shift(1).rolling(window=window).mean()
            data[f'{target_col}_Vol_{window}'] = data[target_col].shift(1).rolling(window=window).std()
        return data
    
    @staticmethod
    def create_lag_features(data, target_col='WTI_Crude', other_features=None, lags=[1, 2, 3, 5, 7]):
        """
        Creates lag features for target and other specified features.
        
        Args:
            data (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            other_features (list): Other feature column names
            lags (list): Lag periods
            
        Returns:
            pd.DataFrame: Dataframe with lag features
        """
        data = data.copy()
        
        # Create lags for target column
        for lag in lags:
            data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        # Create lags for other features
        if other_features is not None:
            for feature in other_features:
                if feature in data.columns:
                    for lag in lags[:3]:  # Use only first 3 lags for other features
                        data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
        return data
    
    @staticmethod
    def create_time_features(data, date_col):
        """
        Creates time-based features from date column.
        
        Args:
            data (pd.DataFrame): Input dataframe
            date_col (str): Date column name
            
        Returns:
            pd.DataFrame: Dataframe with time features
        """
        data = data.copy()
        data['year'] = data[date_col].dt.year
        data['month'] = data[date_col].dt.month
        data['day_of_year'] = data[date_col].dt.dayofyear
        data['weekday'] = data[date_col].dt.dayofweek
        return data


class ModelBacktester:
    """Handles realistic backtesting framework for multiple models."""
    
    def __init__(self, data, target_col='WTI_Crude', date_col='Date'):
        """
        Initialize the backtester.
        
        Args:
            data (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            date_col (str): Date column name
        """
        self.data = data.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.results = {}
        self.feature_engineer = FeatureEngineer()
    
    def prepare_features_historical_only(self, data, p_lags, p_window_sizes):
        """
        Prepares features using only historical data.
        
        Args:
            data (pd.DataFrame): Input dataframe
            p_lags (list): Lag periods
            p_window_sizes (list): Window sizes
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        data = self.feature_engineer.create_time_features(data, self.date_col)
        data = self.feature_engineer.create_technical_features(data, self.target_col, window_sizes=p_window_sizes)
        
        other_features = ['DXY', 'Gold', 'EUR_USD', 'Brent_Oil', 'RBOB_Gasoline', 'Heating_Oil']
        data = self.feature_engineer.create_lag_features(data, self.target_col, other_features, lags=p_lags)
        
        return data
    
    def get_feature_columns(self, data):
        """
        Gets feature columns excluding target and current-day features.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            list: List of feature column names
        """
        exclude_cols = [self.target_col, self.date_col]
        current_day_features = [
            'DXY', 'Gold', 'EUR_USD', 'Brent_Oil', 'RBOB_Gasoline',
            'Heating_Oil', 'Energy_ETF', 'Emerging_Markets', 'FTSE100',
            'Materials', 'CAD_USD', 'NOK_USD', 'Silver', 'Copper'
        ]
        exclude_cols.extend(current_day_features)
        
        feature_cols = [
            col for col in data.columns 
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])
        ]
        return feature_cols
    
    def get_models(self, seasonal_period):
        """
        Returns dictionary of models to use in backtesting.
        
        Args:
            seasonal_period (int): Seasonal period for time series models
            
        Returns:
            dict: Dictionary of model names and instances
        """
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'Prophet': 'prophet_placeholder',
            'ARIMA': 'arima_placeholder',
            'SARIMA': 'sarima_placeholder'
        }
    
    def walk_forward_validation_realistic(self, initial_train_periods, total_test_periods, 
                                        p_lags, p_window_sizes, seasonal_period):
        """
        Performs realistic walk-forward validation.
        
        Args:
            initial_train_periods (int): Initial training periods
            total_test_periods (int): Total test periods
            p_lags (list): Lag periods
            p_window_sizes (list): Window sizes
            seasonal_period (int): Seasonal period
            
        Returns:
            pd.DataFrame: Results dataframe
        """
        print(f"Starting realistic walk-forward validation for {total_test_periods} periods...")
        
        models = self.get_models(seasonal_period)
        all_predictions = {name: [] for name in models.keys()}
        all_actuals, all_dates = [], []
        start_idx = initial_train_periods
        
        # Model orders
        arima_order = (5, 1, 1)
        sarima_order = (2, 1, 2)
        sarima_seasonal_order = (1, 1, 0, seasonal_period)
        
        for i in tqdm(range(total_test_periods), desc="Backtesting"):
            if start_idx + 1 > len(self.data):
                break
                
            train_end_idx = start_idx
            test_end_idx = start_idx + 1
            
            full_featured_data = self.prepare_features_historical_only(
                self.data.iloc[:test_end_idx], p_lags, p_window_sizes
            )
            train_features = full_featured_data.iloc[:train_end_idx].dropna()
            test_features_row = full_featured_data.iloc[start_idx:test_end_idx]
            
            if len(train_features) < 20:
                start_idx += 1
                continue
            
            feature_cols = self.get_feature_columns(train_features)
            X_train = train_features[feature_cols].ffill().bfill()
            y_train = train_features[self.target_col]
            X_test = test_features_row[feature_cols].ffill().bfill()
            y_actual = self.data.iloc[start_idx:test_end_idx][self.target_col].iloc[0]
            
            for name, model in models.items():
                try:
                    if name == 'Prophet':
                        prediction = self._predict_prophet(train_features, test_features_row, feature_cols)
                    elif name == 'ARIMA':
                        prediction = self._predict_arima(y_train, X_train, X_test, arima_order)
                    elif name == 'SARIMA':
                        prediction = self._predict_sarima(y_train, X_train, X_test, 
                                                        sarima_order, sarima_seasonal_order)
                    else:
                        model.fit(X_train, y_train)
                        prediction = model.predict(X_test)[0]
                    
                    all_predictions[name].append(prediction)
                except Exception as e:
                    all_predictions[name].append(np.nan)
            
            all_actuals.append(y_actual)
            all_dates.append(self.data.iloc[start_idx:test_end_idx][self.date_col].iloc[0])
            start_idx += 1
        
        results_df = pd.DataFrame({
            'Date': all_dates, 
            'Actual': all_actuals, 
            **all_predictions
        })
        self.results = results_df
        return self.results
    
    def _predict_prophet(self, train_features, test_features_row, feature_cols):
        """Helper method for Prophet prediction."""
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        prophet_train_df = train_features[[self.date_col, self.target_col] + feature_cols].rename(
            columns={self.date_col: 'ds', self.target_col: 'y'}
        )
        for col in feature_cols:
            m.add_regressor(col)
        m.fit(prophet_train_df)
        
        future_df = test_features_row[[self.date_col] + feature_cols].rename(
            columns={self.date_col: 'ds'}
        )
        future_df[feature_cols] = future_df[feature_cols].ffill().bfill()
        forecast = m.predict(future_df)
        return forecast['yhat'].iloc[0]
    
    def _predict_arima(self, y_train, X_train, X_test, order):
        """Helper method for ARIMA prediction."""
        mod = sm.tsa.statespace.sarimax.SARIMAX(
            y_train, exog=X_train, order=order,
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = mod.fit(disp=False)
        return res.forecast(steps=1, exog=X_test).iloc[0]
    
    def _predict_sarima(self, y_train, X_train, X_test, order, seasonal_order):
        """Helper method for SARIMA prediction."""
        mod = sm.tsa.statespace.sarimax.SARIMAX(
            y_train, exog=X_train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = mod.fit(disp=False)
        return res.forecast(steps=1, exog=X_test).iloc[0]
    
    def calculate_metrics(self):
        """
        Calculates performance metrics for all models.
        
        Returns:
            pd.DataFrame: Performance metrics dataframe
        """
        metrics = {}
        for col in self.results.columns:
            if col not in ['Date', 'Actual']:
                temp_df = pd.DataFrame({
                    'Actual': self.results['Actual'],
                    'Predictions': self.results[col]
                }).dropna()
                
                if len(temp_df) < 2:
                    continue
                
                actual = temp_df['Actual']
                predictions = temp_df['Predictions']
                
                mae = mean_absolute_error(actual, predictions)
                r2 = r2_score(actual, predictions)
                
                # Directional accuracy
                actual_dir = np.diff(actual) > 0
                pred_dir = np.diff(predictions) > 0
                dir_acc = np.mean(actual_dir == pred_dir) * 100
                
                metrics[col] = {
                    'MAE': mae, 
                    'R²': r2, 
                    'Directional_Accuracy': dir_acc
                }
        
        return pd.DataFrame(metrics).T


class ModelForecaster:
    """Handles future forecasting for different model types."""
    
    def __init__(self, target_col='WTI_Crude', date_col='Date'):
        """
        Initialize the forecaster.
        
        Args:
            target_col (str): Target column name
            date_col (str): Date column name
        """
        self.target_col = target_col
        self.date_col = date_col
        self.feature_engineer = FeatureEngineer()
    
    def get_sklearn_models(self):
        """
        Returns dictionary of sklearn models.
        
        Returns:
            dict: Dictionary of model names and instances
        """
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
    
    def forecast_sklearn_model(self, model, full_data, last_known_data, n_periods, 
                             freq, p_lags, p_window_sizes):
        """
        Generates iterative forecast for sklearn models.
        
        Args:
            model: Trained sklearn model
            full_data (pd.DataFrame): Full historical data with features
            last_known_data (pd.DataFrame): Recent data for feature engineering
            n_periods (int): Number of periods to forecast
            freq (str): Frequency ('D', 'W', 'M')
            p_lags (list): Lag periods
            p_window_sizes (list): Window sizes
            
        Returns:
            pd.DataFrame: Forecast dataframe
        """
        print(f"Generating forecast for the next {n_periods} periods...")
        
        # Get feature columns
        exclude_cols = [self.target_col, self.date_col]
        current_day_features = [
            'DXY', 'Gold', 'EUR_USD', 'Brent_Oil', 'RBOB_Gasoline',
            'Heating_Oil', 'Energy_ETF', 'Emerging_Markets', 'FTSE100',
            'Materials', 'CAD_USD', 'NOK_USD', 'Silver', 'Copper'
        ]
        exclude_cols.extend(current_day_features)
        feature_cols = [
            col for col in full_data.columns 
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(full_data[col])
        ]
        
        future_data = last_known_data.copy()
        predictions = []
        last_date = future_data[self.date_col].max()
        
        original_exo_cols = [col for col in current_day_features if col in future_data.columns]
        
        # Date offset mapping
        freq_map = {'D': 'days', 'W': 'weeks', 'M': 'months'}
        offset_kwarg = {freq_map[freq]: 1}
        
        for i in tqdm(range(n_periods), desc="Forecasting"):
            data_for_features = future_data.copy()
            
            # Apply feature engineering
            data_for_features = self.feature_engineer.create_time_features(data_for_features, self.date_col)
            data_for_features = self.feature_engineer.create_technical_features(
                data_for_features, self.target_col, window_sizes=p_window_sizes
            )
            data_for_features = self.feature_engineer.create_lag_features(
                data_for_features, self.target_col, original_exo_cols, lags=p_lags
            )
            
            features_for_prediction = data_for_features.iloc[-1:]
            X_future = features_for_prediction[feature_cols].ffill()
            
            # Handle missing values
            if X_future.isnull().values.any():
                last_historical_features = full_data[feature_cols].iloc[-1]
                X_future = X_future.fillna(last_historical_features)
            
            prediction = model.predict(X_future)[0]
            predictions.append(prediction)
            
            # Create next row
            next_date = last_date + pd.DateOffset(**{freq_map[freq]: i + 1})
            new_row = {self.date_col: next_date, self.target_col: prediction}
            
            last_values = future_data.iloc[-1]
            for col in original_exo_cols:
                new_row[col] = last_values[col]
            
            future_data = pd.concat([future_data, pd.DataFrame([new_row])], ignore_index=True)
        
        forecast_df = pd.DataFrame({
            self.date_col: pd.to_datetime([
                last_date + pd.DateOffset(**{freq_map[freq]: i + 1}) 
                for i in range(n_periods)
            ]),
            'Forecast': predictions
        })
        
        return forecast_df
    
    def forecast_prophet_model(self, full_featured_data, feature_cols, forecast_periods, resample_frequency):
        """
        Generates forecast using Prophet model.
        
        Args:
            full_featured_data (pd.DataFrame): Full historical data with features
            feature_cols (list): Feature column names
            forecast_periods (int): Number of periods to forecast
            resample_frequency (str): Resampling frequency
            
        Returns:
            pd.DataFrame: Forecast dataframe
        """
        print("Re-training Prophet on the full historical dataset...")
        final_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        prophet_full_df = full_featured_data[[self.date_col, self.target_col] + feature_cols].rename(
            columns={self.date_col: 'ds', self.target_col: 'y'}
        )
        
        for col in feature_cols:
            final_model.add_regressor(col)
        final_model.fit(prophet_full_df)
        
        print(f"Generating forecast for the next {forecast_periods} periods...")
        future_dates = final_model.make_future_dataframe(periods=forecast_periods, freq=resample_frequency)
        
        # Use last known values for regressors
        X_full = full_featured_data[feature_cols]
        last_known_regressors = X_full.iloc[-1:].to_dict(orient='records')[0]
        for col, val in last_known_regressors.items():
            future_dates[col] = val
        
        forecast = final_model.predict(future_dates.tail(forecast_periods))
        future_predictions_df = forecast[['ds', 'yhat']].rename(
            columns={'ds': self.date_col, 'yhat': 'Forecast'}
        )
        
        return future_predictions_df


class DataExporter:
    """Handles data export operations for Power BI compatibility."""
    
    @staticmethod
    def structure_data_for_powerbi(df, backtest_results, future_forecasts, 
                                 target_col='WTI_Crude', date_col='Date', 
                                 exclude_models=['ARIMA', 'SARIMA']):
        """
        Structures data for Power BI with proper columns.
        
        Args:
            df (pd.DataFrame): Historical data
            backtest_results (pd.DataFrame): Backtest results
            future_forecasts (dict): Dictionary of future forecasts by model
            target_col (str): Target column name
            date_col (str): Date column name
            exclude_models (list): Models to exclude from export
            
        Returns:
            pd.DataFrame: Combined structured data
        """
        # Historical actuals
        historical_data = df[[date_col, target_col]].copy()
        historical_data['DataType'] = 'Historical Actual'
        historical_data['Model'] = 'Actual'
        historical_data['Value'] = historical_data[target_col]
        historical_data = historical_data[[date_col, 'DataType', 'Model', 'Value']]
        
        # Backtest predictions (exclude specified models)
        backtest_results_filtered = backtest_results.copy()
        for model in exclude_models:
            if model in backtest_results_filtered.columns:
                backtest_results_filtered = backtest_results_filtered.drop(columns=[model])
        
        backtest_melted = backtest_results_filtered.melt(
            id_vars=[date_col, 'Actual'], var_name='Model', value_name='Value'
        )
        backtest_melted['DataType'] = 'Backtest Prediction'
        backtest_melted = backtest_melted[[date_col, 'DataType', 'Model', 'Value']].dropna()
        
        # Future forecasts (exclude specified models)
        structured_future_forecasts = []
        for model_name, forecast_df in future_forecasts.items():
            if model_name not in exclude_models and not forecast_df.empty:
                forecast_df_structured = forecast_df.copy()
                forecast_df_structured['DataType'] = 'Future Forecast'
                forecast_df_structured['Model'] = model_name
                forecast_df_structured = forecast_df_structured.rename(columns={'Forecast': 'Value'})
                structured_future_forecasts.append(
                    forecast_df_structured[[date_col, 'DataType', 'Model', 'Value']].dropna()
                )
        
        # Combine all data
        if structured_future_forecasts:
            combined_data = pd.concat(
                [historical_data, backtest_melted] + structured_future_forecasts, 
                ignore_index=True
            )
        else:
            combined_data = pd.concat([historical_data, backtest_melted], ignore_index=True)
        
        combined_data.dropna(subset=['Value'], inplace=True)
        return combined_data
    
    @staticmethod
    def export_to_csv(data, filename):
        """
        Exports data to CSV file.
        
        Args:
            data (pd.DataFrame): Data to export
            filename (str): Output filename
        """
        data.to_csv(filename, index=False)
        print(f"Exported data to '{filename}'")


class Visualizer:
    """Handles visualization operations."""
    
    @staticmethod
    def plot_results(df, target_col, date_col, forecast_start_date=None, title="Oil Price Analysis"):
        """
        Plots historical data, backtest results, and forecasts.
        
        Args:
            df (pd.DataFrame): Combined data with DataType column
            target_col (str): Target column name
            date_col (str): Date column name
            forecast_start_date (datetime): Date when forecast starts
            title (str): Plot title
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(18, 8))
        
        # Separate data types
        historical_actuals = df[df['DataType'] == 'Historical Actual'].copy()
        backtest_predictions = df[df['DataType'] == 'Backtest Prediction'].copy()
        future_forecast = df[df['DataType'] == 'Future Forecast'].copy()
        
        # Plot historical actuals
        if not historical_actuals.empty:
            plt.plot(historical_actuals[date_col], historical_actuals['Value'], 
                    label='Historical Actual', color='royalblue', linewidth=2)
        
        # Plot backtest predictions
        if not backtest_predictions.empty:
            unique_models_backtest = backtest_predictions['Model'].unique()
            colors = sns.color_palette("husl", len(unique_models_backtest))
            color_map = dict(zip(unique_models_backtest, colors))
            
            for model_name in unique_models_backtest:
                model_data = backtest_predictions[backtest_predictions['Model'] == model_name]
                plt.plot(model_data[date_col], model_data['Value'], 
                        label=f'{model_name} Backtest', color=color_map[model_name], 
                        linestyle=':', linewidth=1.5)
        
        # Plot future forecasts
        if not future_forecast.empty:
            unique_models_forecast = future_forecast['Model'].unique()
            forecast_colors = sns.color_palette("tab10", len(unique_models_forecast))
            forecast_color_map = dict(zip(unique_models_forecast, forecast_colors))
            
            for model_name in unique_models_forecast:
                model_data = future_forecast[future_forecast['Model'] == model_name]
                plt.plot(model_data[date_col], model_data['Value'], 
                        label=f'{model_name} Forecast', color=forecast_color_map[model_name], 
                        linestyle='--', marker='o', markersize=4)
        
        # Add forecast start line
        if forecast_start_date:
            plt.axvline(forecast_start_date, color='red', linestyle=':', linewidth=2, 
                       label='Forecast Start')
        
        plt.title(title, fontsize=18, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()