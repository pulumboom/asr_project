from typing import Dict

import torch
from torch import nn
from torch.nn.functional import log_softmax


class ConvBlock(nn.Module):
    def __init__(self, conv_layers, activation):
        super().__init__()

        self.model = nn.Sequential()
        for i in range(len(conv_layers)):
            self.model.append(conv_layers[i])
            self.model.append(nn.BatchNorm2d(conv_layers[i].out_channels))
            self.model.append(eval(activation))

    def forward(self, specs):
        return self.model(specs)


class RNNBlock(nn.Module):
    def __init__(self, rnn_layers, activation, device):
        super().__init__()

        self.rnn_layers = []
        self.bn_layers = []
        self.activations = []
        for i in range(len(rnn_layers)):
            self.rnn_layers.append(rnn_layers[i])
            self.bn_layers.append(nn.BatchNorm1d(
                rnn_layers[i].hidden_size * (2 if self.rnn_layers[0].bidirectional else 1)
            ))
            self.activations.append(eval(activation))
        self.rnn_layers = nn.ModuleList(self.rnn_layers)
        self.bn_layers = nn.ModuleList(self.bn_layers)
        self.activations = nn.ModuleList(self.activations)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    def forward(self, specs):
        # hidden_state = torch.zeros(
        #     self.rnn_layers[0].num_layers * (2 if self.rnn_layers[0].bidirectional else 1),
        #     specs.shape[0],
        #     self.rnn_layers[0].hidden_size,
        # ).to(self.device)
        hidden_state = None
        for i in range(len(self.rnn_layers)):
            specs, hidden_state = self.rnn_layers[i](specs, hidden_state)
            specs = self.bn_layers[i](specs.transpose(-1, -2)).transpose(-1, -2)
            specs = self.activations[i](specs)

        return specs


class DeepSpeechV2(nn.Module):
    def __init__(
            self,
            cnn_layers: list[nn.Module],
            rnn_layers: list[nn.Module],
            activation: nn.Module,
            n_tokens: int,
            device
    ):
        super().__init__()
        self.cnn_layers = ConvBlock(cnn_layers, activation)
        self.rnn_layers = RNNBlock(rnn_layers, activation, device)
        self.fc = nn.Linear(rnn_layers[-1].hidden_size * (2 if rnn_layers[-1].bidirectional else 1), n_tokens)

    def forward(self, **batch):
        specs = batch["spectrogram"]
        specs = self.cnn_layers(specs)
        specs = specs.view(specs.shape[0], -1, specs.shape[-1]).transpose(-1, -2)
        specs = self.rnn_layers(specs)
        specs = self.fc(specs)
        log_probs = log_softmax(specs, dim=-1)
        batch["log_probs"] = log_probs
        batch["log_probs_length"] = torch.tensor([log_probs.shape[-2]] * log_probs.shape[0])
        return batch
