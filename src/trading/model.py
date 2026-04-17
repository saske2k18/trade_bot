"""
Multi-Timeframe LSTM Model - Нейронная сеть для обучения на множественных таймфреймах.
Поддерживает 5 криптовалют и 7 таймфреймов (5m, 15m, 1h, 4h, 12h, 1d, 1w).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import yaml
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MultiTimeframeLSTM(nn.Module):
    """
    LSTM модель с поддержкой множественных таймфреймов.
    Использует attention механизм для объединения признаков с разных таймфреймов.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 num_timeframes: int = 7, dropout: float = 0.2, 
                 fusion_method: str = 'attention'):
        """
        Инициализация модели.
        
        Args:
            input_size: Количество признаков на одном таймфрейме
            hidden_size: Размер скрытого слоя LSTM
            num_layers: Количество слоев LSTM
            num_timeframes: Количество таймфреймов
            dropout: Dropout rate
            fusion_method: Метод объединения ('attention', 'concat', 'weighted_average')
        """
        super(MultiTimeframeLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_timeframes = num_timeframes
        self.fusion_method = fusion_method
        self.dropout_rate = dropout
        
        # LSTM для каждого таймфрейма (будет инициализировано в forward при известном количестве)
        self.lstm_layers = None
        self.num_timeframes_actual = num_timeframes
        
        # Attention механизм для объединения таймфреймов
        if fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attention_query = nn.Linear(hidden_size, hidden_size)
        
        # Fully connected слои для прогноза (размер будет определен динамически)
        self.fc_input_size = None
        self.fc1 = None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = None
    
    def _init_layers(self, actual_num_timeframes: int):
        """Инициализация слоев при известном количестве таймфреймов."""
        if self.lstm_layers is not None:
            return
            
        self.num_timeframes_actual = actual_num_timeframes
        
        # LSTM для каждого таймфрейма
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout.p if self.num_layers > 1 else 0
            ) for _ in range(actual_num_timeframes)
        ])
        
        # Определение размера входа для FC слоя
        if self.fusion_method == 'concat':
            self.fc_input_size = self.hidden_size * actual_num_timeframes
        else:
            self.fc_input_size = self.hidden_size
        
        self.fc1 = nn.Linear(self.fc_input_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, 1)
        
        # Перемещение на устройство
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        self.lstm_layers.to(device)
        if self.fusion_method == 'attention':
            self.attention.to(device)
            self.attention_query.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            x: Список тензоров [batch, seq_len, features] для каждого таймфрейма
            
        Returns:
            Прогноз [batch, 1]
        """
        # Инициализация слоев при первом проходе
        actual_num_timeframes = len(x)
        self._init_layers(actual_num_timeframes)
        
        # Применяем LSTM к каждому таймфрейму
        lstm_outputs = []
        for i, lstm in enumerate(self.lstm_layers):
            output, (hidden, cell) = lstm(x[i])
            # Берем последний выход LSTM
            lstm_outputs.append(output[:, -1, :])  # [batch, hidden_size]
        
        # Stack всех выходов: [batch, num_timeframes, hidden_size]
        stacked = torch.stack(lstm_outputs, dim=1)
        
        # Fusion
        if self.fusion_method == 'attention':
            # Создаем query из среднего по таймфреймам
            mean_features = stacked.mean(dim=1, keepdim=True)  # [batch, 1, hidden_size]
            query = self.attention_query(mean_features)
            
            # Attention
            attended, _ = self.attention(query, stacked, stacked)
            fused = attended.squeeze(1)  # [batch, hidden_size]
        
        elif self.fusion_method == 'concat':
            fused = stacked.view(stacked.size(0), -1)  # [batch, num_timeframes * hidden_size]
        
        else:  # weighted_average
            weights = torch.softmax(torch.ones(self.num_timeframes_actual), dim=0).to(stacked.device)
            fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)  # [batch, hidden_size]
        
        # Fully connected layers
        out = self.fc1(fused)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class MultiCryptoTrainer:
    """Тренер для обучения модели на множестве криптовалют и таймфреймов."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Инициализация тренера."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.symbols = self.config['trading']['symbols']
        self.timeframes = self.config['trading']['timeframes']
        self.model_config = self.config['model']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Скалеры для нормализации данных (отдельный для каждой крипты)
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Модели (отдельная для каждой крипты или одна общая)
        self.models: Dict[str, MultiTimeframeLSTM] = {}
        
        logger.info(f"MultiCryptoTrainer initialized for {len(self.symbols)} symbols")
    
    def prepare_sequences(self, df: pd.DataFrame, lookback: int, 
                          target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка последовательностей для LSTM.
        
        Args:
            df: DataFrame с признаками
            lookback: Длина истории (lookback window)
            target_col: Целевая колонка для прогноза
            
        Returns:
            X: [samples, lookback, features]
            y: [samples, 1]
        """
        # Целевая переменная: направление движения цены (1 - вверх, 0 - вниз)
        y = (df[target_col].shift(-1) > df[target_col]).astype(int).values
        
        # Удаляем последнюю строку (для нее нет целевого значения)
        df = df.iloc[:-1]
        y = y[:-1]
        
        # Нормализация данных
        feature_cols = [c for c in df.columns if c not in ['symbol', 'timeframe']]
        scaler = StandardScaler()
        df_normalized = df.copy()
        df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # Сохраняем скалер
        scaler_name = f"default"
        self.scalers[scaler_name] = scaler
        
        # Создание последовательностей
        X, y_seq = [], []
        for i in range(len(df_normalized) - lookback):
            X.append(df_normalized[feature_cols].iloc[i:i+lookback].values)
            y_seq.append(y[i + lookback - 1])
        
        return np.array(X), np.array(y_seq)
    
    def prepare_multi_timeframe_data(self, data_dict: Dict[str, pd.DataFrame], 
                                     symbol: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Подготовка данных для всех таймфреймов.
        
        Args:
            data_dict: {timeframe: DataFrame} для одной криптовалюты
            symbol: Название криптовалюты
            
        Returns:
            Список (X_tf, y_tf) для каждого таймфрейма
        """
        lookback_config = self.model_config['lookback']
        prepared_data = []
        
        for tf in self.timeframes:
            if tf not in data_dict:
                logger.warning(f"No data for timeframe {tf}")
                continue
            
            df = data_dict[tf]
            lookback = lookback_config.get(tf, 50)
            
            X, y = self.prepare_sequences(df, lookback)
            prepared_data.append((X, y, tf))
            
            logger.info(f"Prepared {len(X)} sequences for {symbol} {tf} (lookback={lookback})")
        
        return prepared_data
    
    def create_model(self, input_size: int) -> MultiTimeframeLSTM:
        """Создание модели согласно конфигурации."""
        model = MultiTimeframeLSTM(
            input_size=input_size,
            hidden_size=self.model_config['hidden_units'],
            num_layers=self.model_config.get('num_layers', 2),
            num_timeframes=len(self.timeframes),
            dropout=self.model_config['dropout_rate'],
            fusion_method=self.model_config.get('fusion_method', 'attention')
        ).to(self.device)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def train_model(self, data_dict: Dict[str, pd.DataFrame], symbol: str, 
                    epochs: int = None, batch_size: int = None) -> dict:
        """
        Обучение модели для одной криптовалюты.
        
        Args:
            data_dict: {timeframe: DataFrame}
            symbol: Название криптовалюты
            epochs: Количество эпох
            batch_size: Размер батча
            
        Returns:
            История обучения
        """
        epochs = epochs or self.model_config['epochs']
        batch_size = batch_size or self.model_config['batch_size']
        
        logger.info(f"Training model for {symbol}")
        
        # Подготовка данных
        prepared_data = self.prepare_multi_timeframe_data(data_dict, symbol)
        
        if len(prepared_data) == 0:
            raise ValueError(f"No data prepared for {symbol}")
        
        # Получаем размер признаков из первого таймфрейма
        sample_X, _, _ = prepared_data[0]
        input_size = sample_X.shape[2]
        
        # Создание модели
        model = self.create_model(input_size)
        self.models[symbol] = model
        
        # Оптимизатор и функция потерь
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Разделение на train/val
        train_ratio = self.model_config['train_split']
        val_ratio = self.model_config['validation_split']
        
        # Объединяем данные всех таймфреймов
        # Для простоты берем минимальное количество сэмплов across all timeframes
        min_samples = min(len(X) for X, _, _ in prepared_data)
        
        train_end = int(min_samples * train_ratio)
        val_end = int(min_samples * (train_ratio + val_ratio))
        
        # Подготовка DataLoader
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            # Перемешивание индексов
            indices = np.random.permutation(train_end)
            
            for i in range(0, train_end, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Подготовка батча для всех таймфреймов
                batch_x = []
                batch_y = None
                
                for X, y, _ in prepared_data:
                    X_batch = torch.FloatTensor(X[batch_indices]).to(self.device)
                    if batch_y is None:
                        batch_y = torch.FloatTensor(y[batch_indices]).to(self.device)
                    batch_x.append(X_batch)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(-1), batch_y)  # squeeze только последнее измерение
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for i in range(train_end, val_end, batch_size):
                    batch_indices = list(range(i, min(i+batch_size, val_end)))
                    
                    batch_x = []
                    batch_y = None
                    
                    for X, y, _ in prepared_data:
                        X_batch = torch.FloatTensor(X[batch_indices]).to(self.device)
                        if batch_y is None:
                            batch_y = torch.FloatTensor(y[batch_indices]).to(self.device)
                        batch_x.append(X_batch)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(-1), batch_y)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        logger.info(f"Training complete for {symbol}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model': model
        }
    
    def train_all_symbols(self, all_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, dict]:
        """
        Обучение моделей для всех криптовалют.
        
        Args:
            all_data: {symbol: {timeframe: DataFrame}}
            
        Returns:
            Результаты обучения для каждой криптовалюты
        """
        results = {}
        
        for symbol in self.symbols:
            if symbol not in all_data:
                logger.warning(f"No data for symbol {symbol}, skipping")
                continue
            
            try:
                result = self.train_model(all_data[symbol], symbol)
                results[symbol] = result
                
                # Сохранение модели
                self.save_model(symbol)
                
            except Exception as e:
                logger.error(f"Error training {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def save_model(self, symbol: str, path: str = "models/"):
        """Сохранение модели."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        if symbol in self.models:
            model_path = f"{path}/{symbol.replace('/', '_')}_model.pth"
            torch.save(self.models[symbol].state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
    
    def load_model(self, symbol: str, path: str = "models/") -> MultiTimeframeLSTM:
        """Загрузка обученной модели."""
        model_path = f"{path}/{symbol.replace('/', '_')}_model.pth"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Определение input_size из конфига (упрощенно)
        input_size = 30  # Примерное количество признаков
        
        model = self.create_model(input_size)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        self.models[symbol] = model
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def predict(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> float:
        """
        Прогноз для последней доступной точки данных.
        
        Args:
            symbol: Криптовалюта
            data_dict: {timeframe: DataFrame} с последними данными
            
        Returns:
            Вероятность роста цены (0-1)
        """
        if symbol not in self.models:
            self.load_model(symbol)
        
        model = self.models[symbol]
        model.eval()
        
        # Подготовка данных (аналогично обучению)
        prepared_data = self.prepare_multi_timeframe_data(data_dict, symbol)
        
        if len(prepared_data) == 0:
            raise ValueError("No data for prediction")
        
        # Берем последние последовательности
        batch_x = []
        for X, _, _ in prepared_data:
            X_last = torch.FloatTensor(X[-1:]).to(self.device)
            batch_x.append(X_last)
        
        with torch.no_grad():
            output = model(batch_x)
            probability = torch.sigmoid(output).item()
        
        logger.info(f"Prediction for {symbol}: {probability:.4f} ({'UP' if probability > 0.5 else 'DOWN'})")
        return probability


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO)
    
    # Создание тестовых данных
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
    
    test_data = {
        'BTC/USDT': {}
    }
    
    for tf in ['1h', '4h', '1d']:
        test_df = pd.DataFrame({
            'open': np.random.randn(500).cumsum() + 100,
            'high': np.random.randn(500).cumsum() + 101,
            'low': np.random.randn(500).cumsum() + 99,
            'close': np.random.randn(500).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
        test_data['BTC/USDT'][tf] = test_df
    
    # Добавление признаков
    import sys
    sys.path.insert(0, '/workspace')
    from src.data.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    
    for tf in test_data['BTC/USDT']:
        test_data['BTC/USDT'][tf] = fe.add_all_features(test_data['BTC/USDT'][tf])
    
    # Обучение
    trainer = MultiCryptoTrainer()
    
    print("\nStarting training...")
    results = trainer.train_model(test_data['BTC/USDT'], 'BTC/USDT', epochs=20, batch_size=16)
    
    print(f"\nTraining complete!")
    print(f"Final train loss: {results['train_losses'][-1]:.4f}")
    print(f"Final val loss: {results['val_losses'][-1]:.4f}")
