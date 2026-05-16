# Gold and Silver Price Prediction

This project trains price prediction models for gold and silver using historical data.

## Files
- `gold_silver_predictor.py`: main Python script for training, evaluation, prediction, and model saving.
- `streamlit_app.py`: Streamlit web app for interactive prediction and deployment.
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

Run the Streamlit app locally:
```bash
streamlit run streamlit_app.py
```

## Deployment
This repository is ready for deployment on Streamlit Community Cloud.
1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud and create a new app.
3. Connect your GitHub repo and set the main branch.
4. Deploy the app.

## Notes
- The app automatically detects gold and silver columns by searching for `gold` or `silver` in the dataset headers.
- The dataset file is not included in this repository, so you must provide `Gold-Silver-Data.csv`.
