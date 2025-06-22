import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.missing_strategies = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.columns.tolist()
        
        for col in columns:
            if col not in df_copy.columns:
                continue
                
            if strategy == 'mean' and df_copy[col].dtype in ['int64', 'float64']:
                fill_value = df_copy[col].mean()
            elif strategy == 'median' and df_copy[col].dtype in ['int64', 'float64']:
                fill_value = df_copy[col].median()
            elif strategy == 'mode':
                fill_value = df_copy[col].mode().iloc[0] if not df_copy[col].mode().empty else 0
            elif strategy == 'forward_fill':
                df_copy[col] = df_copy[col].fillna(method='ffill')
                continue
            elif strategy == 'backward_fill':
                df_copy[col] = df_copy[col].fillna(method='bfill')
                continue
            else:
                fill_value = 0
            
            df_copy[col] = df_copy[col].fillna(fill_value)
            self.missing_strategies[col] = {'strategy': strategy, 'fill_value': fill_value}
        
        return df_copy
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
        df_copy = df.copy()
        
        for col in columns:
            if col not in df_copy.columns:
                continue
            
            if method == 'onehot':
                dummies = pd.get_dummies(df_copy[col], prefix=col)
                df_copy = pd.concat([df_copy.drop(col, axis=1), dummies], axis=1)
                self.encoders[col] = {'method': 'onehot', 'columns': dummies.columns.tolist()}
            
            elif method == 'label':
                unique_values = df_copy[col].unique()
                label_map = {val: idx for idx, val in enumerate(unique_values)}
                df_copy[col] = df_copy[col].map(label_map)
                self.encoders[col] = {'method': 'label', 'mapping': label_map}
        
        return df_copy
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        df_copy = df.copy()
        
        for col in columns:
            if col not in df_copy.columns:
                continue
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                continue
            
            df_copy[col] = scaler.fit_transform(df_copy[[col]]).flatten()
            self.scalers[col] = scaler
        
        return df_copy
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        df_copy = df.copy()
        
        for col in columns:
            if col not in df_copy.columns or df_copy[col].dtype not in ['int64', 'float64']:
                continue
            
            if method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
            
            elif method == 'z_score':
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                df_copy = df_copy[z_scores < threshold]
        
        return df_copy


def create_sample_dataset(n_samples: int = 1000, n_features: int = 5, noise: float = 0.1, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_state:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # Create a linear relationship with some noise
    true_weights = np.random.randn(n_features)
    y = np.dot(X, true_weights) + noise * np.random.randn(n_samples)
    
    return X, y


def create_classification_dataset(n_samples: int = 1000, n_features: int = 2, n_classes: int = 3, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_state:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # Create class centers
    centers = np.random.randn(n_classes, n_features) * 3
    
    # Assign samples to classes based on distance to centers
    distances = np.array([np.linalg.norm(X - center, axis=1) for center in centers])
    y = np.argmin(distances, axis=0)
    
    # Add some noise to X
    X += 0.5 * np.random.randn(n_samples, n_features)
    
    return X, y


def generate_time_series(n_points: int = 100, trend: float = 0.01, seasonality: float = 0.5, noise: float = 0.1, random_state: Optional[int] = None) -> np.ndarray:
    if random_state:
        np.random.seed(random_state)
    
    t = np.arange(n_points)
    
    # Trend component
    trend_component = trend * t
    
    # Seasonality component
    seasonal_component = seasonality * np.sin(2 * np.pi * t / 12)
    
    # Noise component
    noise_component = noise * np.random.randn(n_points)
    
    return trend_component + seasonal_component + noise_component