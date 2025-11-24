Advanced Time Series Forecasting with Seq2Seq + Multi-Head Attention
------------------------------------------------------------------
This improved project provides:
- data_loader.py : data loading & preprocessing (reads data/electricity.csv if present; falls back to synthetic data)
- model.py       : Seq2Seq architecture using PyTorch with multi-head attention
- train.py       : Training loop, checkpoint saving, rolling-origin CV helper
- baseline.py    : Simple SARIMA baseline wrapper (requires statsmodels)
- evaluate.py    : MAE, RMSE, MAPE metrics + save attention weights
- utils.py       : helper functions (scaler save/load)
- requirements.txt : suggested packages
- example_usage.txt : how to run
Notes: Place a CSV at data/electricity.csv with a datetime index and a target column named 'load' to use real data.
