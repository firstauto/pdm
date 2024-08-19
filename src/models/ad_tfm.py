from turtle import xcor
import torch
import torch.nn as nn


class AD_TFM(nn.Module):
    def __init__(
        self,
        d_model,
        emb_size=1024,
        nhead=8,
        seq_len=16,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        activation=None,
        dropout=0.1,
        norm_first=False,
    ):
        super(AD_TFM, self).__init__()
        # self.emb_in = nn.Linear(d_model, emb_size, bias=False)
        # self.emb_out = nn.Linear(d_model, emb_size, bias=False)

        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(
                    d_model,
                    nhead,
                    seq_len,
                    dim_feedforward,
                    activation,
                    dropout,
                    norm_first,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                TransformerDecoder(
                    d_model,
                    nhead,
                    seq_len,
                    dim_feedforward,
                    activation,
                    dropout,
                    norm_first,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        # self.output_layer = nn.Linear(emb_size, d_model, bias=False)

        encoder_embedding = nn.Parameter(
            torch.rand((1, d_model, seq_len), requires_grad=True) * 2 - 1
        )
        self.register_parameter("encoder_embedding", encoder_embedding)
        decoder_embedding = nn.Parameter(
            torch.rand((1, d_model, seq_len), requires_grad=True) * 2 - 1
        )
        self.register_parameter("decoder_embedding", decoder_embedding)

    def forward(self, src, tgt):
        memory = self.encoder_embedding + src.permute(0, 2, 1)
        for layer in self.encoder:
            memory = layer(memory)
        output = self.decoder_embedding + tgt.permute(0, 2, 1)
        for layer in self.decoder:
            output = layer(output, memory)

        # output = self.output_layer(output)
        return output.permute(0, 2, 1)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        seq_len,
        dim_feedforward,
        activation,
        dropout,
        norm_first=False,
    ):
        super(TransformerEncoder, self).__init__()
        self.norm = norm_first
        self.self_attn = nn.MultiheadAttention(
            seq_len, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = ConvLayer(d_model, dim_feedforward, activation, dropout=dropout)
        self.norm1 = nn.LayerNorm(seq_len)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src2 = src
        if self.norm:
            src2 = self.norm1(src)
        src2 = self.dropout(self.self_attn(src2, src2, src2)[0])
        if not self.norm:
            src2 = self.norm1(src2)
        src = src + src2
        src2 = src
        if self.norm:
            src2 = self.norm2(src2)
        src = src + self.ffn(src2)
        if not self.norm:
            src = self.norm2(src)
        return src


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        seq_len,
        dim_feedforward,
        activation,
        dropout,
        norm_first=False,
    ):
        super(TransformerDecoder, self).__init__()
        self.norm = norm_first
        self.self_attn = nn.MultiheadAttention(
            seq_len, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            seq_len, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = ConvLayer(d_model, dim_feedforward, activation, dropout=dropout)
        self.norm1 = nn.LayerNorm(seq_len)
        self.norm2 = nn.LayerNorm(seq_len)
        self.norm3 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt2 = tgt
        if self.norm:
            tgt2 = self.norm1(tgt2)
        tgt2 = self.dropout3(self.self_attn(tgt2, tgt2, tgt2)[0])
        if not self.norm:
            tgt2 = self.norm1(tgt2)
        tgt = tgt + tgt2
        tgt2 = tgt
        if self.norm:
            tgt2 = self.norm2(tgt2)
        tgt2 = self.dropout4(self.multihead_attn(tgt2, memory, memory)[0])
        if not self.norm:
            tgt2 = self.norm2(tgt2)
        tgt = tgt + tgt2
        tgt2 = tgt
        if self.norm:
            tgt2 = self.norm3(tgt2)
        tgt = tgt + self.ffn(tgt2)
        if not self.norm:
            tgt = self.norm3(tgt)
        return tgt


class ConvLayer(nn.Module):
    def __init__(self, d_model, d_ff, activation, kernel_size=1, dropout=0.0):
        super(ConvLayer, self).__init__()
        self.activation = activation
        self.kernel_size_ = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size + 2
        self.conv1 = nn.Conv1d(
            d_model,
            d_ff,
            kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.conv2 = nn.Conv1d(
            d_ff, d_model, self.kernel_size_, padding=(self.kernel_size_ - 1) // 2
        )
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        x = x
        x = self.activation(self.conv1(x))
        x = self.conv2(self.dropout1(x))
        return x