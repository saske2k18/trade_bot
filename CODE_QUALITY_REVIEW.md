# Codebase Review: Errors, Inconsistencies, and Improvement Plan

## What was checked

- Core training pipeline: `src/main.py`
- Data ingestion: `src/data/data_collector.py`
- Feature engineering: `src/data/feature_engineering.py`
- Model and trainer: `src/trading/model.py`
- Config and docs consistency: `configs/config.yaml`, `README.md`, `TRAINING_GUIDE.md`, `CLOUD_TRAINING_GUIDE.md`

---

## Key issues found

### 1) Model loading could fail due to hardcoded `input_size`
- `load_model()` previously created the network with a fixed `input_size = 30`.
- In practice, the number of engineered features can differ, causing checkpoint shape mismatch.
- Impact: inability to load trained models reliably in production.

### 2) `num_timeframes` mismatch risk between training and loading
- The model dynamically initializes LSTM branches from actual timeframe data used in a run.
- Previously, load path relied on config defaults and not saved metadata.
- Impact: shape mismatch or incorrect architecture during restore.

### 3) Training loop edge cases could produce invalid training state
- If dataset is too short relative to lookback/splits, training could run with zero batches.
- Impact: runtime errors (or NaN behavior), brittle UX.

### 4) Resampling logic in multi-timeframe feature merge was inconsistent
- Fallback frequency was always `'1H'` even for target timeframes like `5m`, `4h`, `1d`, `1w`.
- Impact: feature alignment errors and potential target/feature temporal distortion.

### 5) Documentation vs actual code structure mismatch
- `README.md` references modules (`executor.py`, `strategy.py`, etc.) that are absent.
- Impact: onboarding friction and confusion for contributors.

---

## Changes made now

1. **Checkpoint saving/loading made robust**
   - Save metadata with model weights (`input_size`, `num_timeframes`, and core hyperparameters).
   - Load uses metadata-first path; legacy checkpoints still supported with warning fallback.

2. **Model creation aligned with actual available timeframes**
   - `create_model()` now accepts explicit `num_timeframes`.
   - Training path passes the real number of prepared timeframes.

3. **Training safety guards added**
   - Explicit validation when split leaves no training batches.
   - More actionable errors/warnings for too-small datasets.

4. **Correct timeframe resampling in feature engineering**
   - Added map from exchange timeframes (`5m`, `1h`, `1d`, `1w`, etc.) to pandas frequencies.
   - Resampling now uses target timeframe frequency instead of hardcoded `'1H'` fallback.

---

## Recommended next improvements (high impact)

### A) Prevent subtle leakage in scaling
Current sequence prep fits scaler before train/validation split. Better:
- Fit scaler on train window only.
- Apply to val/test/predict with same scaler.
- Store scalers by `(symbol, timeframe)`.

### B) Add strict temporal validation protocol
For time series:
- Use walk-forward validation (rolling windows) instead of single split.
- Track directional metrics: F1 for UP/DOWN, precision@top-confidence, calibration.

### C) Improve label quality
Current label is binary next-candle direction. Consider:
- Triple-barrier labeling or return-threshold labeling to suppress noise.
- Cost-aware target that includes spread/fees/slippage.

### D) Expand tests (currently minimal)
Add unit tests for:
- Feature outputs and NaN handling.
- Sequence preparation dimensions and split behavior.
- Save/load roundtrip with non-default feature sizes.

### E) Reproducibility and observability
- Add global seeds (`python`, `numpy`, `torch`).
- Log dataset sizes and class balance per symbol/timeframe.
- Save training config snapshot with each model artifact.

### F) Align docs with code
- Update `README.md` structure and commands to match actual modules.
- Add a concise “production checklist” (data freshness, latency budget, retraining cadence).

---

## Suggested roadmap

1. **Stability sprint (1–2 days):** tests + scaler discipline + seed control.
2. **Model quality sprint (3–5 days):** walk-forward validation + improved labeling.
3. **Trading realism sprint (3–5 days):** transaction-cost-aware objective and backtest metrics.
4. **MLOps sprint (2–3 days):** experiment tracking (MLflow/W&B), model registry, scheduled retraining.

