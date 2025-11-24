import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        x = self.input_linear(x)
        outputs, (h, c) = self.rnn(x)
        return outputs, (h, c)

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_heads=4, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=False)
        self.rnn_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, decoder_inputs, hidden=None):
        batch = encoder_outputs.size(0)
        enc_len = encoder_outputs.size(1)
        dec_len = decoder_inputs.size(1)
        memory = encoder_outputs.permute(1,0,2)
        outputs = []
        if hidden is None:
            h_t = torch.zeros(batch, self.hidden_size, device=encoder_outputs.device)
            c_t = torch.zeros(batch, self.hidden_size, device=encoder_outputs.device)
        else:
            h_t, c_t = hidden
        attn_weights = None
        for t in range(dec_len):
            query = h_t.unsqueeze(0)
            attn_output, attn_weights = self.attn(query=query, key=memory, value=memory, need_weights=True)
            attn_output = attn_output.squeeze(0)
            h_t, c_t = self.rnn_cell(attn_output, (h_t, c_t))
            out = self.output_layer(h_t)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, attn_weights
