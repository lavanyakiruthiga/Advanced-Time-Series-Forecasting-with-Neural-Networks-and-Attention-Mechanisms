import numpy as np

def mae(preds, y):
    return float(np.mean(np.abs(preds - y)))

def rmse(preds, y):
    return float(np.sqrt(np.mean((preds - y)**2)))

def mape(preds, y, eps=1e-8):
    return float(np.mean(np.abs((y - preds) / (y + eps))) * 100.0)

def save_attention_weights(attn, path):
    try:
        import numpy as np
        np.save(path, attn)
        print(f'Attention weights saved to {path}')
    except Exception as e:
        print('Failed to save attention weights:', e)
