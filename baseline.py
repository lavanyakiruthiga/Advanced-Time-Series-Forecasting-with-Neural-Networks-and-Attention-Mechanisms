# Simple baseline wrapper using SARIMAX (statsmodels)
def sarima_train_predict(train_series, steps=24, order=(1,0,1), seasonal_order=(0,1,1,24)):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as e:
        raise RuntimeError('statsmodels is required for SARIMA baseline: pip install statsmodels') from e
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.forecast(steps=steps)
    return pred
