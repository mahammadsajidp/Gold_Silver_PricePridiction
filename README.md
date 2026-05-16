# Gold and Silver Price Prediction

This project trains price prediction models for gold and silver using historical data.

## Files
- `gold_silver_predictor.py`: main Python script for training, evaluation, prediction, and model saving.
- `requirements.txt`: Python dependencies.
- `.gitignore`: ignores Python artifacts and model files.

## Setup
1. Place your dataset `Gold-Silver-Data.csv` in the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Train and evaluate a model for gold or silver:
```bash
python gold_silver_predictor.py --target gold
```

To predict after training:
```bash
python gold_silver_predictor.py --target gold --predict
```

To save the trained model:
```bash
python gold_silver_predictor.py --target silver --save-model
```

## Notes
- The script automatically detects gold and silver columns by searching for `gold` or `silver` in the dataset headers.
- The dataset file is not included in this repository, so you must provide `Gold-Silver-Data.csv`.
