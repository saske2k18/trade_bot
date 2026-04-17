"""
Feature Engineering - Создание технических индикаторов и признаков для обучения модели.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging
import yaml

logger = logging.getLogger(__name__)


TIMEFRAME_TO_PANDAS_FREQ = {
    '1m': '1min',
    '3m': '3min',
    '5m': '5min',
    '15m': '15min',
    '30m': '30min',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h',
    '6h': '6h',
    '8h': '8h',
    '12h': '12h',
    '1d': '1D',
    '3d': '3D',
    '1w': '1W',
}


class FeatureEngineer:
    """Класс для создания технических индикаторов и признаков."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Инициализация инженера признаков."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.features_config = self.config.get('data', {}).get('features', [])
        logger.info("FeatureEngineer initialized")
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление базовых признаков: доходности, лог-доходности."""
        df = df.copy()
        
        # Доходность (returns)
        df['return'] = df['close'].pct_change()
        
        # Лог-доходность
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Волатильность (rolling standard deviation)
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['return'].rolling(window=window).std()
        
        # Скользящие средние цены
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        logger.debug(f"Added basic features. Shape: {df.shape}")
        return df
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Добавление RSI (Relative Strength Index)."""
        df = df.copy()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        logger.debug(f"Added RSI({period})")
        return df
    
    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                 signal: int = 9) -> pd.DataFrame:
        """Добавление MACD (Moving Average Convergence Divergence)."""
        df = df.copy()
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        logger.debug(f"Added MACD({fast}, {slow}, {signal})")
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                            std_dev: float = 2.0) -> pd.DataFrame:
        """Добавление полос Боллинджера."""
        df = df.copy()
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        logger.debug(f"Added Bollinger Bands({period}, {std_dev})")
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление признаков объема."""
        df = df.copy()
        
        # Скользящая средняя объема
        for window in [5, 10, 20]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Отношение текущего объема к среднему
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Накопленный объем (OBV-like)
        df['volume_cumsum'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        logger.debug("Added volume features")
        return df
    
    def add_price_position(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Позиция цены относительно минимума/максимума за период."""
        df = df.copy()
        
        rolling_high = df['high'].rolling(window=lookback).max()
        rolling_low = df['low'].rolling(window=lookback).min()
        
        df['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low)
        
        logger.debug(f"Added price position feature (lookback={lookback})")
        return df
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление всех признаков согласно конфигурации."""
        logger.info("Adding all features to dataset")
        
        # Базовые признаки
        df = self.add_basic_features(df)
        
        # RSI
        df = self.add_rsi(df)
        
        # MACD
        df = self.add_macd(df)
        
        # Bollinger Bands
        df = self.add_bollinger_bands(df)
        
        # Volume features
        df = self.add_volume_features(df)
        
        # Price position
        df = self.add_price_position(df)
        
        # Удаление NaN значений
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}, dropped {dropped} NaN rows")
        return df
    
    def prepare_multi_timeframe_data(self, data_dict: Dict[str, pd.DataFrame], 
                                     target_timeframe: str = '1h') -> pd.DataFrame:
        """
        Подготовка данных для мульти-таймфрейм модели.
        Объединение признаков с разных таймфреймов в один DataFrame.
        
        Args:
            data_dict: Словарь {timeframe: DataFrame}
            target_timeframe: Целевой таймфрейм для прогноза
            
        Returns:
            DataFrame с объединенными признаками со всех таймфреймов
        """
        logger.info(f"Preparing multi-timeframe data with target: {target_timeframe}")
        
        # Базовый DataFrame - целевой таймфрейм
        if target_timeframe not in data_dict:
            raise ValueError(f"Target timeframe {target_timeframe} not found in data")
        
        base_df = data_dict[target_timeframe].copy()
        base_df = self.add_all_features(base_df)
        
        # Переименование колонок с префиксом таймфрейма
        base_cols = [c for c in base_df.columns if c not in ['symbol', 'timeframe']]
        rename_dict = {c: f"{target_timeframe}_{c}" for c in base_cols}
        base_df.rename(columns=rename_dict, inplace=True)
        
        # Добавление признаков с других таймфреймов
        for tf, df in data_dict.items():
            if tf == target_timeframe:
                continue
            
            # Добавляем признаки
            df_featured = self.add_all_features(df.copy())
            
            # Берем только ключевые признаки для избежания дублирования
            key_features = ['close', 'return', 'rsi', 'macd', 'bb_pct', 'volume_ratio']
            available_features = [f for f in key_features if f in df_featured.columns]
            
            # Ресемплинг к целевому таймфрейму (если нужно)
            if len(df_featured) > 0:
                # Группировка по времени и взятие последнего значения
                target_freq = TIMEFRAME_TO_PANDAS_FREQ.get(target_timeframe, '1h')
                df_resampled = df_featured[available_features].resample(target_freq).last()
                
                # Переименование с префиксом таймфрейма
                rename_dict = {c: f"{tf}_{c}" for c in available_features}
                df_resampled.rename(columns=rename_dict, inplace=True)
                
                # Объединение
                base_df = base_df.join(df_resampled, how='inner')
        
        logger.info(f"Multi-timeframe data prepared. Final shape: {base_df.shape}")
        return base_df


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO)
    
    # Создаем тестовые данные
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    test_df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    fe = FeatureEngineer()
    featured_df = fe.add_all_features(test_df)
    
    print("\nFeatured DataFrame:")
    print(featured_df.tail())
    print(f"\nShape: {featured_df.shape}")
    print(f"Columns: {list(featured_df.columns)}")
