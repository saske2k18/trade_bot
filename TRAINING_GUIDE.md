# Руководство по обучению нейросети на криптовалютах

## 📋 Описание

Данная система позволяет обучать нейронную сеть (Multi-Timeframe LSTM) для прогнозирования движения цен криптовалют. 

**Поддерживаемые криптовалюты (топ-5):**
- BTC/USDT (Bitcoin)
- ETH/USDT (Ethereum)
- BNB/USDT (Binance Coin)
- SOL/USDT (Solana)
- XRP/USDT (Ripple)

**Поддерживаемые таймфреймы:**
- 5m (5 минут)
- 15m (15 минут)
- 1h (1 час)
- 4h (4 часа)
- 12h (12 часов)
- 1d (1 день)
- 1w (1 неделя)

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
cd /workspace
pip install -r requirements.txt
```

### 2. Настройка API ключей (опционально)

Скопируйте `.env.example` в `.env` и добавьте ваши API ключи Binance:

```bash
cp .env.example .env
```

Отредактируйте `.env`:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

> ⚠️ Для загрузки исторических данных API ключи не обязательны (можно использовать публичный API).

### 3. Запуск обучения

#### Базовый запуск (используя настройки из конфига):

```bash
python src/main.py
```

#### Продвинутый запуск с параметрами:

```bash
# Обучение только BTC и ETH на часовом и дневном таймфреймах
python src/main.py --symbols BTC/USDT ETH/USDT --timeframes 1h 1d --epochs 50

# Обучение всех 5 криптовалют с ограничением в 500 свечей
python src/main.py --limit 500 --epochs 30
```

### 4. Просмотр результатов

После обучения модели сохраняются в папку `models/`:
- `BTC_USDT_model.pth`
- `ETH_USDT_model.pth`
- и т.д.

Логи обучения сохраняются в `logs/training.log`.

## 📁 Структура проекта

```
trade_bot/
├── configs/
│   └── config.yaml          # Конфигурация (символы, таймфреймы, параметры модели)
├── src/
│   ├── main.py              # Точка входа для обучения
│   ├── data/
│   │   ├── data_collector.py       # Загрузка данных с биржи
│   │   └── feature_engineering.py  # Создание технических индикаторов
│   └── trading/
│       └── model.py         # Multi-Timeframe LSTM модель
├── models/                  # Сохраненные модели (.pth файлы)
├── data/
│   ├── raw/                # Сырые данные (кэш)
│   └── cache/              # Обработанные данные
├── logs/                   # Логи обучения
└── requirements.txt        # Зависимости
```

## ⚙️ Конфигурация

Файл `configs/config.yaml` содержит все настройки:

### Торговые настройки
```yaml
trading:
  symbols:
    - "BTC/USDT"
    - "ETH/USDT"
    - "BNB/USDT"
    - "SOL/USDT"
    - "XRP/USDT"
  timeframes:
    - "5m"
    - "15m"
    - "1h"
    - "4h"
    - "12h"
    - "1d"
    - "1w"
```

### Параметры модели
```yaml
model:
  type: "multi_timeframe_lstm"
  lookback:
    "5m": 60    # 5 часов истории
    "15m": 48   # 12 часов истории
    "1h": 48    # 2 дня истории
    "4h": 48    # 8 дней истории
    "12h": 20   # 10 дней истории
    "1d": 30    # 1 месяц истории
    "1w": 12    # 3 месяца истории
  hidden_units: 128
  num_layers: 2
  dropout_rate: 0.2
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
  fusion_method: "attention"  # attention, concat, weighted_average
```

## 🧠 Архитектура модели

### Multi-Timeframe LSTM

Модель использует отдельный LSTM для каждого таймфрейма с последующим объединением через:

1. **Attention механизм** (по умолчанию) - динамически определяет важность каждого таймфрейма
2. **Concat** - конкатенация выходов всех LSTM
3. **Weighted Average** - взвешенное среднее

### Технические индикаторы

Автоматически рассчитываются для каждого таймфрейма:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Скользящие средние (SMA, EMA)
- Волатильность
- Объемные индикаторы

Всего ~33 признака на каждый таймфрейм.

## 📊 Пример использования в коде

```python
from src.data.data_collector import DataCollector
from src.data.feature_engineering import FeatureEngineer
from src.trading.model import MultiCryptoTrainer

# 1. Сбор данных
collector = DataCollector()
data = collector.get_latest_candles("BTC/USDT", "1h", count=500)

# 2. Добавление признаков
fe = FeatureEngineer()
featured_data = fe.add_all_features(data)

# 3. Обучение
trainer = MultiCryptoTrainer()
result = trainer.train_model(
    data_dict={"1h": featured_data},
    symbol="BTC/USDT",
    epochs=30
)

# 4. Прогноз
probability = trainer.predict("BTC/USDT", {"1h": featured_data})
print(f"Вероятность роста: {probability:.2%}")
```

## 🔍 Интерпретация результатов

- **Train Loss < 0.5**: Модель обучается хорошо
- **Val Loss ≈ Train Loss**: Нет переобучения
- **Val Loss >> Train Loss**: Переобучение (увеличьте dropout или уменьшите epochs)

Прогноз модели:
- **> 0.5**: Ожидается рост цены
- **< 0.5**: Ожидается падение цены
- **≈ 0.5**: Неопределенность

## ⚠️ Важные замечания

1. **Качество данных**: Чем больше исторических данных, тем лучше обучение
2. **Время обучения**: Обучение на всех 7 таймфреймах может занять 1-2 часа
3. **Ресурсы**: Рекомендуется использовать GPU для ускорения обучения
4. **Риски**: Это экспериментальная система, не используйте для реальной торговли без дополнительного тестирования

## 🛠️ Troubleshooting

### Ошибка "No data received"
- Проверьте интернет-соединение
- Уменьшите количество запрашиваемых свечей (`--limit`)
- Некоторые таймфреймы могут быть недоступны для новых монет

### Ошибка "CUDA out of memory"
- Уменьшите `batch_size` в конфиге
- Уменьшите `hidden_units`
- Отключите GPU: установите `CUDA_VISIBLE_DEVICES=""`

### Медленное обучение
- Используйте меньше таймфреймов
- Уменьшите `lookback` периоды
- Включите GPU

## 📝 Лицензия

MIT License - свободное использование с указанием авторства.

## 🤝 Вклад в проект

Pull requests приветствуются! Пожалуйста, убедитесь, что:
- Код проходит тесты
- Добавлены соответствующие тесты
- Обновлена документация
