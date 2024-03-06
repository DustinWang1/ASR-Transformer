import torch.nn as nn
import torch
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel feature, time)


# ResCNN cell
class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = nn.functional.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = nn.functional.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x

class SpeechRecognitionModel(nn.Module):
    def __init__(self, vocab_size: int = 31, n_cnn_layers: int = 4, d_model: int = 512, n_feats: int = 128, stride: int = 1, dropout: float = 0.1):
        super().__init__()

        #vocab size is num characters
        self.label_embedding = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride,
                             padding=3 // 2)
        self.resCNNLayers = nn.Sequential(*[
            ResCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])

        self.FC = nn.Linear(n_feats*32, d_model)

        self.transformer = nn.Transformer(d_model=d_model, nhead=8, batch_first=True)

        self.projection = nn.Linear(d_model, vocab_size)

    def encode(self, spec):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = self.cnn(spec)
        x = self.resCNNLayers(x)  # (batch, channel, n_feats, seq_len)

        # Prepare for transformer
        # (batch, seq_len, d_model) d_model = 512
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, seq, d_model)
        x = self.FC(x)
        x = self.transformer.encoder(x)
        return x

    def decode(self, encoder_output, decoder_input):
        decoder_input = self.label_embedding(decoder_input)
        x = self.transformer.decoder(decoder_input, encoder_output)
        x = self.projection(x)
        return x

    def forward(self, spec, label):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = self.cnn(spec)
        x = self.resCNNLayers(x)  # (batch, channel, n_feats, seq_len)

        # Prepare for transformer
        # (batch, seq_len, d_model) d_model = 512
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, seq, d_model)

        x = self.FC(x)
        # Input Embedding for label
        label = self.label_embedding(label)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(label.size(1)).to(device)
        x = self.transformer(src=x, tgt=label, tgt_mask=causal_mask)

        # Output of transformer is (batch, seq, embedding)
        return self.projection(x)  # (batch, seq, vocab_size)



def build_transformer():

    return SpeechRecognitionModel()




# What special characters for tokenization?
# Add extra ResCNN?
# CTCLoss?
