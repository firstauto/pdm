import torch
import torch.nn as nn
import numpy as np

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, seq_len, dropout):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True), num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True), num_decoder_layers
        )
        self.embedding = nn.Linear(num_features, d_model)
        self.output_layer = nn.Linear(d_model, num_features)

        encoder_embedding = nn.Parameter(
            torch.rand((1, seq_len, d_model), requires_grad=True) * 2 - 1
        )
        self.register_parameter("encoder_embedding", encoder_embedding)
        decoder_embedding = nn.Parameter(
            torch.rand((1, seq_len, d_model), requires_grad=True) * 2 - 1
        )
        self.register_parameter("decoder_embedding", decoder_embedding)

    def forward(self, src, tgt):

        src, tgt = src.permute(0, 2, 1), tgt.permute(0, 2, 1)

        src = self.embedding(src) * torch.sin(self.encoder_embedding)
        tgt = self.embedding(tgt) * torch.sin(self.decoder_embedding)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.output_layer(output)
        return output.permute(0, 2, 1)
