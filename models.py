import torch
import numpy as np
from torch import nn


class BaselineDNN(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, pooling='mean'):

        if pooling not in ['mean', 'mean_max']:
            raise ValueError("Invalid pooling method.")

        super(BaselineDNN, self).__init__()
        self.hiddel_size = 100
        self.pooling = pooling
        _, dim = np.array(embeddings).shape

        if pooling == 'mean_max':
            dim *= 2

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
        self.embedding.weight.requires_grad = trainable_emb

        self.linear = nn.Linear(dim, self.hidden_size)
        self.relu = nn.ReLU()

        self.output = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, lengths):
        
        embeddings = self.embedding(x)

        if self.pooling == 'mean':
            representations = embeddings.sum(dim=1) / lengths.float().unsqueeze(1)
        elif self.pooling == 'mean_max':
            mean_rep = embeddings.sum(dim=1) / lengths.float().unsqueeze(1)
            max_rep = embeddings.max(dim=1).values
            representations = torch.cat([mean_rep, max_rep], dim=1)

        representations = self.linear(representations)
        representations = self.relu(representations)

        logits = self.output(representations)

        return logits


class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)
        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)

        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)
        representations = ...

        logits = self.linear(representations)

        return logits
