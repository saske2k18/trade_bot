"""
Main Training Script - Запуск обучения нейросети на 5 криптовалютах и 7 таймфреймах.
"""

import logging
import argparse
from pathlib import Path
import yaml

from src.data.data_collector import DataCollector
from src.data.feature_engineering import FeatureEngineer
from src.trading.model import MultiCryptoTrainer


def setup_logging(config_path: str = "configs/config.yaml"):
    """Настройка логирования."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('file', 'logs/training.log')
    
    # Создание директории для логов
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Основная функция обучения."""
    parser = argparse.ArgumentParser(description='Обучение ML модели для криптовалют')
    parser.add_argument('--symbols', nargs='+', help='Список криптовалют для обучения')
    parser.add_argument('--timeframes', nargs='+', help='Список таймфреймов')
    parser.add_argument('--epochs', type=int, help='Количество эпох обучения')
    parser.add_argument('--limit', type=int, default=1000, help='Количество свечей для загрузки')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Путь к конфигу')
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging(args.config)
    logger.info("=" * 60)
    logger.info("Starting cryptocurrency ML model training")
    logger.info("=" * 60)
    
    # Загрузка конфига
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    symbols = args.symbols or config['trading']['symbols']
    timeframes = args.timeframes or config['trading']['timeframes']
    
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    
    # Шаг 1: Сбор данных
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Collecting data from exchange")
    logger.info("=" * 60)
    
    collector = DataCollector(config_path=args.config)
    
    # Обновляем символы и таймфреймы в коллекторе
    collector.symbols = symbols
    collector.timeframes = timeframes
    
    all_data = {}
    for symbol in symbols:
        logger.info(f"\nCollecting data for {symbol}...")
        all_data[symbol] = {}
        
        for tf in timeframes:
            try:
                df = collector.get_latest_candles(symbol, tf, count=args.limit)
                
                if not df.empty:
                    all_data[symbol][tf] = df
                    logger.info(f"  ✓ {tf}: {len(df)} candles")
                else:
                    logger.warning(f"  ✗ {tf}: No data received")
                    
            except Exception as e:
                logger.error(f"  ✗ {tf}: Error - {str(e)}")
    
    # Проверка наличия данных
    valid_symbols = [s for s in symbols if len(all_data.get(s, {})) > 0]
    
    if not valid_symbols:
        logger.error("No data collected! Exiting.")
        return
    
    logger.info(f"\nData collection complete for {len(valid_symbols)} symbols")
    
    # Шаг 2: Feature Engineering
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Feature engineering")
    logger.info("=" * 60)
    
    feature_engineer = FeatureEngineer(config_path=args.config)
    
    for symbol in valid_symbols:
        logger.info(f"\nProcessing features for {symbol}...")
        
        for tf in all_data[symbol]:
            df = all_data[symbol][tf]
            
            # Добавление всех признаков
            featured_df = feature_engineer.add_all_features(df)
            all_data[symbol][tf] = featured_df
            
            logger.info(f"  ✓ {tf}: {len(featured_df)} samples, {featured_df.shape[1]} features")
    
    # Шаг 3: Обучение моделей
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Training models")
    logger.info("=" * 60)
    
    trainer = MultiCryptoTrainer(config_path=args.config)
    
    epochs = args.epochs or config['model']['epochs']
    
    results = {}
    for symbol in valid_symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Training model for {symbol}")
        logger.info(f"{'='*40}")
        
        try:
            result = trainer.train_model(
                data_dict=all_data[symbol],
                symbol=symbol,
                epochs=epochs
            )
            
            results[symbol] = {
                'status': 'success',
                'train_loss': result['train_losses'][-1] if result['train_losses'] else None,
                'val_loss': result['val_losses'][-1] if result['val_losses'] else None
            }
            
            logger.info(f"✓ Training complete for {symbol}")
            logger.info(f"  Final train loss: {result['train_losses'][-1]:.4f}")
            logger.info(f"  Final val loss: {result['val_losses'][-1]:.4f}")
            
        except Exception as e:
            logger.error(f"✗ Training failed for {symbol}: {str(e)}")
            results[symbol] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Итоговый отчет
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    error_count = sum(1 for r in results.values() if r['status'] == 'error')
    
    logger.info(f"Total symbols: {len(valid_symbols)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {error_count}")
    
    for symbol, result in results.items():
        if result['status'] == 'success':
            logger.info(f"  ✓ {symbol}: train_loss={result['train_loss']:.4f}, val_loss={result['val_loss']:.4f}")
        else:
            logger.info(f"  ✗ {symbol}: {result['error']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training pipeline completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
