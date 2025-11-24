import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import Encoder, AttentionDecoder
from data_loader import load_csv, generate_synthetic, prepare_sequences, train_val_test_split, to_torch
from evaluate import mae, rmse, mape, save_attention_weights
import argparse

def build_model(input_size, hidden_size, output_size, num_heads=4, device='cpu'):
    encoder = Encoder(input_size=input_size, hidden_size=hidden_size).to(device)
    decoder = AttentionDecoder(hidden_size=hidden_size, output_size=output_size, num_heads=num_heads).to(device)
    return encoder, decoder

def train_epoch(encoder, decoder, dataloader, optimizer, criterion, device):
    encoder.train(); decoder.train()
    total_loss = 0.0
    for xb, yb in dataloader:
        xb = xb.to(device); yb = yb.to(device)
        optimizer.zero_grad()
        enc_outs, (h,c) = encoder(xb)
        dec_inputs = torch.zeros(yb.size(0), yb.size(1), xb.size(2), device=device)
        preds, attn = decoder(enc_outs, dec_inputs)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_model(encoder, decoder, dataloader, device):
    encoder.eval(); decoder.eval()
    preds_all = []
    ys_all = []
    attn_weights = None
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device); yb = yb.to(device)
            enc_outs, _ = encoder(xb)
            dec_inputs = torch.zeros(yb.size(0), yb.size(1), xb.size(2), device=device)
            preds, attn = decoder(enc_outs, dec_inputs)
            preds_all.append(preds.cpu())
            ys_all.append(yb.cpu())
            attn_weights = attn
    import torch
    preds_all = torch.cat(preds_all, dim=0).numpy()
    ys_all = torch.cat(ys_all, dim=0).numpy()
    return preds_all, ys_all, attn_weights

def main(args):
    if os.path.exists('data/electricity.csv'):
        df = load_csv('data/electricity.csv')
        series = df['load'] if 'load' in df.columns else df.iloc[:,0]
    else:
        print('No dataset found at data/electricity.csv. Generating synthetic dataset for demo.')
        df = generate_synthetic(n=5000, freq=24)
        series = df['load']

    X, Y, scaler = prepare_sequences(series, input_len=args.input_len, output_len=args.output_len)
    (Xtr,Ytr),(Xval,Yval),(Xte,Yte) = train_val_test_split(X,Y,val_frac=0.1,test_frac=0.1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = TensorDataset(to_torch(Xtr), to_torch(Ytr))
    val_ds = TensorDataset(to_torch(Xval), to_torch(Yval))
    test_ds = TensorDataset(to_torch(Xte), to_torch(Yte))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    encoder, decoder = build_model(input_size=1, hidden_size=args.hidden_size, output_size=1, num_heads=args.num_heads, device=device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(encoder, decoder, train_loader, optimizer, criterion, device)
        preds_val, ys_val, _ = evaluate_model(encoder, decoder, val_loader, device)
        v_mae = mae(preds_val, ys_val)
        print(f'Epoch {epoch} | Train Loss: {train_loss:.6f} | Val MAE: {v_mae:.6f}')
        if v_mae < best_val:
            best_val = v_mae
            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, 'checkpoint.pth')
    preds_test, ys_test, attn = evaluate_model(encoder, decoder, test_loader, device)
    print('Test MAE:', mae(preds_test, ys_test))
    print('Test RMSE:', rmse(preds_test, ys_test))
    print('Test MAPE:', mape(preds_test, ys_test))
    save_attention_weights(attn, 'attention_weights.npy')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=48)
    parser.add_argument('--output_len', type=int, default=24)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
