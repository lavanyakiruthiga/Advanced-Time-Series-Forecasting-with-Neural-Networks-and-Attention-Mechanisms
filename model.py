import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (h, c) = self.rnn(x)
        return outputs, (h, c)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.softmax(self.v(energy), dim=1)
        return attention

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs, attention_layer):
        attn_weights = attention_layer(hidden[-1], encoder_outputs)
        context = torch.sum(attn_weights * encoder_outputs, dim=1).unsqueeze(1)
        output, (h, c) = self.rnn(context, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, (h, c)
