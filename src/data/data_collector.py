"""Загрузка исторических данных криптовалют с биржи."""

import ccxt
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DataCollector:
    """Класс для сбора исторических данных с криптобирж."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Инициализация сборщика данных."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.exchange_id = self.config['trading']['exchange']
        self.symbols = self.config['trading']['symbols']
        self.timeframes = self.config['trading']['timeframes']
        
        # Инициализация биржи
        self.exchange = getattr(ccxt, self.exchange_id)({
            'enableRateLimit': True,
            'apiKey': os.getenv('BINANCE_API_KEY', ''),
            'secret': os.getenv('BINANCE_API_SECRET', ''),
        })
        
        # Директории для сохранения данных
        self.base_dir = Path(self.config['data']['cache_dir'])
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataCollector initialized for {len(self.symbols)} symbols and {len(self.timeframes)} timeframes")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000,
                    since: Optional[int] = None) -> pd.DataFrame:
        """
        Загрузка OHLCV данных для указанной пары и таймфрейма.
        
        Args:
            symbol: Торговая пара (например, 'BTC/USDT')
            timeframe: Таймфрейм (например, '1h', '4h', '1d')
            limit: Количество свечей для загрузки
            since: Начальная временная метка (в мс)
            
        Returns:
            DataFrame с OHLCV данными
        """
        try:
            logger.info(f"Fetching {limit} candles for {symbol} on {timeframe}")
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
            
            if not ohlcv:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            logger.info(f"Successfully loaded {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_all_data(self, limit: int = 1000) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Загрузка данных для всех символов и таймфреймов.
        
        Args:
            limit: Количество свечей для каждого таймфрейма
            
        Returns:
            Словарь вида: {symbol: {timeframe: DataFrame}}
        """
        all_data = {}
        
        for symbol in self.symbols:
            all_data[symbol] = {}
            logger.info(f"Processing symbol: {symbol}")
            
            for timeframe in self.timeframes:
                df = self.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not df.empty:
                    all_data[symbol][timeframe] = df
                    
                    # Сохранение в файл
                    self._save_data(symbol, timeframe, df)
                else:
                    logger.warning(f"No data for {symbol} {timeframe}")
        
        logger.info(f"Data collection complete. Loaded data for {len(all_data)} symbols")
        return all_data
    
    def _save_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Сохранение данных в CSV файл."""
        symbol_safe = symbol.replace('/', '_')
        filepath = self.base_dir / f"{symbol_safe}_{timeframe}.csv"
        
        df.to_csv(filepath)
        logger.debug(f"Saved data to {filepath}")
    
    def load_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Загрузка кэшированных данных из файла."""
        symbol_safe = symbol.replace('/', '_')
        filepath = self.base_dir / f"{symbol_safe}_{timeframe}.csv"
        
        if filepath.exists():
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            logger.info(f"Loaded cached data from {filepath}")
            return df
        
        logger.warning(f"Cached data not found: {filepath}")
        return None
    
    def get_latest_candles(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Получение последних N свечей для символа и таймфрейма."""
        df = self.load_cached_data(symbol, timeframe)
        
        if df is not None and len(df) >= count:
            return df.tail(count)
        
        return self.fetch_ohlcv(symbol, timeframe, limit=count)

    def fetch_full_history(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        batch_limit: int = 1000,
        max_batches: Optional[int] = None,
    ) -> pd.DataFrame:
        """Загрузка истории с пагинацией батчами от start_date до текущего момента."""
        start_date = start_date or self.config.get('data', {}).get('history', {}).get('start_date', '2010-01-01')
        since_ms = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
        batches = max_batches or self.config.get('data', {}).get('history', {}).get('max_batches_per_request', 3000)

        logger.info(
            "Fetching full history for %s %s from %s (batch_limit=%s, max_batches=%s)",
            symbol,
            timeframe,
            start_date,
            batch_limit,
            batches,
        )

        all_chunks: List[pd.DataFrame] = []
        last_ts: Optional[int] = None

        for batch_idx in range(batches):
            chunk = self.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=batch_limit, since=since_ms)
            if chunk.empty:
                break

            chunk = chunk.sort_index()
            all_chunks.append(chunk)

            latest_ts = int(chunk.index[-1].timestamp() * 1000)
            if last_ts is not None and latest_ts <= last_ts:
                break

            last_ts = latest_ts
            since_ms = latest_ts + 1

            if len(chunk) < batch_limit:
                break

            if (batch_idx + 1) % 50 == 0:
                logger.info("Fetched %s batches for %s %s", batch_idx + 1, symbol, timeframe)

        if not all_chunks:
            logger.warning("Full history is empty for %s %s", symbol, timeframe)
            return pd.DataFrame()

        df = pd.concat(all_chunks).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        self._save_data(symbol, timeframe, df)
        logger.info("Fetched %s rows of full history for %s %s", len(df), symbol, timeframe)
        return df


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO)
    
    collector = DataCollector()
    
    # Загрузка данных для всех символов и таймфреймов
    print(f"Symbols: {collector.symbols}")
    print(f"Timeframes: {collector.timeframes}")
    
    # Загрузка данных для BTC/USDT на разных таймфреймах
    for tf in collector.timeframes[:3]:  # Первые 3 для теста
        df = collector.get_latest_candles("BTC/USDT", tf, count=50)
        print(f"\n{tf}: {len(df)} candles")
        print(df.tail())
