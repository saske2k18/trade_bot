# 🚀 Гайд: Бесплатное обучение в облаке

Этот гайд описывает, как бесплатно обучить вашу нейросеть на 5 криптовалютах и 7 таймфреймах, используя мощь облачных GPU.

## 📋 Варианты бесплатных облачных сервисов

| Сервис | GPU | Время сессии | Диск | Особенности |
|--------|-----|--------------|------|-------------|
| **Google Colab** | T4 / V100 / A100 | 12 часов | ~80 GB | ⭐ Лучший выбор, есть Drive интеграция |
| **Kaggle Kernels** | P100 / T4 | 9 часов | 20 GB | Требует верификации телефона |
| **Paperspace Gradient** | M4000 / P5000 | 6 часов | 50 GB | Есть постоянные хранилища |
| **Lightning AI** | CPU/GPU | 22 часа/мес | 10 GB | Удобный интерфейс |

---

## 🏆 Рекомендация: Google Colab (Самый простой вариант)

### Шаг 1: Подготовка

1. **Создайте аккаунт Google** (если нет)
2. **Перейдите на [colab.research.google.com](https://colab.research.google.com)**
3. **Откройте предоставленный ноутбук:**
   - Нажмите `File` → `Upload notebook`
   - Загрузите файл `notebooks/train_on_colab.ipynb` из этого репозитория
   - ИЛИ скопируйте ссылку на GitHub и используйте `Open from GitHub`

### Шаг 2: Настройка среды

1. **Включите GPU:**
   ```
   Runtime (Среда выполнения) → Change runtime type → Hardware accelerator: GPU → Save
   ```

2. **Проверьте доступность GPU:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Должно вывести True
   print(torch.cuda.get_device_name(0))  # Покажет модель GPU
   ```

### Шаг 3: Запуск обучения

1. **Подключите Google Drive** (первая ячейка):
   - Следуйте инструкции по авторизации
   - Модели будут сохраняться в папку `crypto_bot_models` на вашем диске

2. **Запустите все ячейки:**
   ```
   Runtime → Run all
   ```
   
   Или поочередно нажимайте `▶️` на каждой ячейке.

### Шаг 4: Мониторинг процесса

- Вы увидите логи загрузки данных для каждой монеты и таймфрейма
- Прогресс обучения по эпохам
- Потери (Loss) будут уменьшаться

### Шаг 5: Сохранение результатов

После обучения модели автоматически сохранятся в:
```
Google Drive → crypto_bot_models/
├── BTC_USDT_model.pth
├── ETH_USDT_model.pth
├── BNB_USDT_model.pth
├── SOL_USDT_model.pth
└── XRP_USDT_model.pth
```

---

## ⚡ Альтернатива: Kaggle Kernels

Если Colab недоступен или закончился лимит:

### Шаг 1: Регистрация
1. Перейдите на [kaggle.com](https://www.kaggle.com)
2. Войдите через Google аккаунт
3. Верифицируйте номер телефона (требуется для доступа к GPU)

### Шаг 2: Создание ноутбука
1. Нажмите `Code` → `New Notebook`
2. В настройках справа выберите:
   - **Accelerator**: GPU
   - **Internet**: On (для скачивания данных)

### Шаг 3: Код для Kaggle

```python
# Установка зависимостей
!pip install ccxt pandas numpy torch scikit-learn --quiet

# Остальной код аналогичен Colab, но сохраняйте модели в:
# /kaggle/working/models/
import os
os.makedirs('/kaggle/working/models', exist_ok=True)

# При сохранении:
torch.save(model.state_dict(), f"/kaggle/working/models/{symbol}_model.pth")
```

### Шаг 4: Скачивание моделей
После обучения:
1. Вкладка `Output` → `models`
2. Скачайте `.pth` файлы кнопкой `Download`

---

## 🔧 Оптимизация для бесплатных тарифов

### 1. Экономия времени
Бесплатные сессии ограничены. Чтобы успеть обучиться:

```yaml
# В config.yaml уменьшите параметры для быстрого старта:
training:
  epochs: 30          # Вместо 100
  batch_size: 64      # Больше = быстрее
  test_split: 0.15    # Меньше тестовых данных
  
timeframes:
  - 1h                # Начните с 1-2 таймфреймов
  - 4h
  # Добавьте остальные позже
```

### 2. Проверка прогресса
Добавьте раннюю остановку в код:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
# В цикле обучения:
scheduler.step(val_loss)
```

### 3. Чекпоинты
Сохраняйте модель каждые 5 эпох:

```python
if (epoch + 1) % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f"checkpoint_{epoch}.pth")
```

---

## 📊 Что делать после обучения

### 1. Локальное тестирование
Скачайте модели и протестируйте на исторических данных:

```bash
python src/backtest.py --model models/BTC_USDT_model.pth
```

### 2. Дообучение
Модели можно дообучать на новых данных:

```python
# Загрузите существующие веса
model.load_state_dict(torch.load('BTC_USDT_model.pth'))
# Продолжите обучение на свежих данных
```

### 3. Деплой
Используйте обученные модели для:
- Бумажной торговли (paper trading)
- Реальных сигналов
- Интеграции с торговым ботом

---

## ⚠️ Важные замечания

### Лимиты бесплатных тарифов

| Сервис | Лимит GPU | Примечания |
|--------|-----------|------------|
| Colab Free | ~12 часов/сессия | Может быть очередь на GPU |
| Colab+ | $10/мес | Приоритетный доступ |
| Kaggle | 30 часов/неделя | Нужно активировать вручную |

### Советы по стабильности

1. **Не закрывайте вкладку браузера** во время обучения
2. **Используйте стабилизатор соединения** (расширения типа "Colab Auto Refresh")
3. **Регулярно сохраняйте чекпоинты** на Google Drive
4. **Начинайте обучение утром** (меньше нагрузка на сервера)

### Если сессия прервалась

1. Данные уже загружены в переменные — можно продолжить
2. Чекпоинты сохранены на диске — загрузите последнюю версию
3. Используйте `resume_training=True` в коде

---

## 🎯 Быстрый старт (Copy-Paste)

Для максимально быстрого запуска в Colab:

```python
# 1. Установить зависимости
!pip install ccxt pandas numpy torch scikit-learn -q

# 2. Подключить Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Склонировать ваш репозиторий
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

# 4. Запустить обучение
!python src/main.py --epochs 30 --symbols BTC/USDT ETH/USDT
```

---

## 📞 Поддержка

Если возникли проблемы:
1. Проверьте логи ошибок в ноутбуке
2. Убедитесь, что GPU активен: `!nvidia-smi`
3. Попробуйте уменьшить размер батча или количество эпох
4. Для Colab: очистите Runtime → `Factory restart runtime`

**Удачи в обучении! 🚀**
