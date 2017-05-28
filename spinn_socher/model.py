"""
    Model for Paraphrase detection

"""

import itertools
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from spinn import SPINN
from sentence_features import sentence_features


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(*size, -1)


class Linear(Bottle, nn.Linear):
    pass


class BatchNorm(Bottle, nn.BatchNorm1d):
    pass


class Feature(nn.Module):
    """ Determines what features to extract from the embeddings """

    def __init__(self, size, dropout):
        super(Feature, self).__init__()
        self.bn = nn.BatchNorm1d(size * 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sentence1, sentence2):
        return self.dropout(self.bn(
            torch.cat(
                [sentence1,                 # plain sentence1 embedding
                 sentence2,                 # plain sentence2 embedding
                 sentence1 - sentence2,     # difference of two vector embeddings
                 sentence1 * sentence2],    # point-wise product of embeddings
                1                           # the dimension to concatenate over
            )
        ))


class Encoder(nn.Module):
    """ LSTM decoder with option for Bi-directional """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                           num_layers=config.n_layers, dropout=config.rnn_dropout,
                           bidirectional=config.birnn)

    def forward(self, inputs, _):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class ParaphraseClassifier(nn.Module):

    def __init__(self, config):
        super(ParaphraseClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.embed_bn = BatchNorm(config.d_proj)
        self.embed_dropout = nn.Dropout(p=config.embed_dropout)
        self.encoder = SPINN(config) if config.spinn else Encoder(config)

        feat_in_size = config.d_hidden * (
            2 if (self.config.birnn and not self.config.spinn) else 1)
        self.feature = Feature(feat_in_size, config.mlp_dropout)

        # Create a multi-layer perceptron to output from encoded representation
        self.mlp_dropout = nn.Dropout(p=config.mlp_dropout)
        self.relu = nn.ReLU()
        # 4 features inside the Feature Layer see Feature class
        mlp_in_size = 4 * feat_in_size
        mlp = [nn.Linear(mlp_in_size, config.d_mlp), self.relu,
               nn.BatchNorm1d(config.d_mlp), self.mlp_dropout]
        for i in range(config.n_mlp_layers - 1):
            mlp.extend([nn.Linear(config.d_mlp, config.d_mlp), self.relu,
                        nn.BatchNorm1d(config.d_mlp), self.mlp_dropout])
        mlp.append(nn.Linear(config.d_mlp, config.d_out))
        mlp.append(nn.LogSoftmax())
        self.out = nn.Sequential(*mlp)

    def forward(self, batch):
        # import pdb
        # pdb.set_trace()

        # get embeddings for each word in sentence
        sentence1_embed = self.embed(batch.sentence1)
        sentence2_embed = self.embed(batch.sentence2)
        if self.config.fix_emb:
            # set the embedding to a Variable to disallow training of it
            sentence1_embed = Variable(sentence1_embed.data)
            sentence2_embed = Variable(sentence2_embed.data)
        # project embedding dimension to another dimension (probably smaller?)
        if self.config.projection:
            sentence1_embed = self.projection(sentence1_embed)  # no relu
            sentence2_embed = self.projection(sentence2_embed)

        sentence1_embed = self.embed_dropout(self.embed_bn(sentence1_embed))
        sentence2_embed = self.embed_dropout(self.embed_bn(sentence2_embed))
        if hasattr(batch, 'sentence1_transitions'):
            sentence1_trans = batch.sentence1_transitions
            sentence2_trans = batch.sentence2_transitions
        else:
            sentence1_trans = sentence2_trans = None

        # encode the sentences using some encoder model
        sentence1 = self.encoder(sentence1_embed, sentence1_trans)
        sentence2 = self.encoder(sentence2_embed, sentence2_trans)
        scores = self.out(self.feature(sentence1, sentence2))
        # print(sentence1[0][:5], sentence2[0][:5])
        return scores


class ParaphraseClassifierExtra(nn.Module):
    """ This is the ParaphraseClassifier with
        some extra features fed into the MLP layer
    """

    def __init__(self, config, sentence_vocab):
        super(ParaphraseClassifierExtra, self).__init__()

        self.sentence_vocab = sentence_vocab
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.embed_bn = BatchNorm(config.d_proj)
        self.embed_dropout = nn.Dropout(p=config.embed_dropout)
        self.encoder = SPINN(config) if config.spinn else Encoder(config)
        self.sim_mat = SimilarityMatrix(pnorm=2)
        self.dynamic_pool = DynamicMin2DPool(grid_size=15)

        feat_in_size = config.d_hidden * (
            2 if self.config.birnn and not self.config.spinn else 1)
        self.feature = Feature(feat_in_size, config.mlp_dropout)

        # The extra features to be added are:

        # Plain Features
        # ===============
        # 1. Difference in sentence length
        # 2. Percentage of words and phrases in one sentence that are in other sentence and vice-versa
        #   a. sentence 1
        #   b. sentence 2
        # Number Features
        # ===============
        # 1. If the two sentences contain exactly the same numbers or no number
        # 2. If two sentences contain the same numbers
        # 3. If the set of numbers of one sentence is the strict subset of the numbers in the other sentence
        n_extra_features = 7    # because Plain Feature (2) gives two features

        # Create a multi-layer perceptron to output from encoded representation
        self.mlp_dropout = nn.Dropout(p=config.mlp_dropout)
        self.relu = nn.ReLU()
        mlp_in_size = 4 * feat_in_size + n_extra_features
        mlp = [nn.Linear(mlp_in_size, config.d_mlp), self.relu,
               nn.BatchNorm1d(config.d_mlp), self.mlp_dropout]
        for i in range(config.n_mlp_layers - 1):
            mlp.extend([nn.Linear(config.d_mlp, config.d_mlp), self.relu,
                        nn.BatchNorm1d(config.d_mlp), self.mlp_dropout])
        mlp.append(nn.Linear(config.d_mlp, config.d_out))
        mlp.append(nn.LogSoftmax())
        self.out = nn.Sequential(*mlp)

    def forward(self, batch):
        # get embeddings for each word in sentence
        sentence1_embed = self.embed(batch.sentence1)
        sentence2_embed = self.embed(batch.sentence2)
        if self.config.fix_emb:
            # set the embedding to a Variable to disallow training of it
            sentence1_embed = Variable(sentence1_embed.data)
            sentence2_embed = Variable(sentence2_embed.data)
        # project embedding dimension to another dimension (probably smaller?)
        if self.config.projection:
            sentence1_embed = self.projection(sentence1_embed)  # no relu
            sentence2_embed = self.projection(sentence2_embed)

        sentence1_embed = self.embed_dropout(self.embed_bn(sentence1_embed))
        sentence2_embed = self.embed_dropout(self.embed_bn(sentence2_embed))
        if hasattr(batch, 'sentence1_transitions'):
            sentence1_trans = batch.sentence1_transitions
            sentence2_trans = batch.sentence2_transitions
        else:
            sentence1_trans = sentence2_trans = None

        # encode the sentences using some encoder model
        sentence1_states = self.encoder(sentence1_embed, sentence1_trans)
        sentence2_states = self.encoder(sentence2_embed, sentence2_trans)

        # replace with SimilarityMatrix here
        sim_mats = self.sim_mat(sentence1_states, sentence2_states)
        # pooling operation here
        pooled_matrix = self.dynamic_pool(sim_mats)

        # finally use softmax as before
        scores = self.out(
            torch.cat(
                [pooled_matrix,
                 sentence_features(batch.sentence1, batch.sentence2, self.sentence_vocab)],
                1   # the dimensions to concatenate over
            )
        )
        return scores


class SimilarityMatrix(nn.Module):

    def __init__(self, p_norm=2):
        super(SimilarityMatrix, self).__init__()

        # self.pdist = nn.PairwiseDistance(p_norm)

    def forward(self, states_1, states_2):
        n, m = len(states_1), len(states_2)

        batch = zip(states_1, states_2)
        simmat_batch = []
        for sent_states_1, sent_states_2 in batch:
            distances = []
            for sent_state_1, sent_state_2 in itertools.product(sent_states_1, sent_states_2):
                distances.append(F.pairwise_distance(sent_state_1, sent_state_2))
            simmat_batch.append(torch.stack(distances, 0).contiguous().view(n, m))
        return simmat_batch


class DynamicMin2DPool(nn.Module):

    def __init__(self, grid_size):
        super(DynamicMin2DPool, self).__init__()
        # the resulting matrix output will be n_p x n_p
        # where the passed grid_size is n_p
        self.grid_size = grid_size

    def forward(self, sim_matrices):
        grid_outputs = []
        for sim_matrix in sim_matrices:
            if sim_matrix.dim() != 2:
                raise ValueError("The sim_matrix passed should be 2-D.")

            n_rows, n_cols = tuple(sim_matrix.size())
            # the dimensions of the pooling kernel
            krows_size = np.floor(n_rows / self.grid_size)
            kcols_size = np.floor(n_cols / self.grid_size)

            n_row_regions = int(np.ceil(n_rows / krows_size))
            n_col_regions = int(np.ceil(n_cols / kcols_size))

            grid_output = Variable(torch.zeros(self.grid_size, self.grid_size))
            for rowi in range(n_row_regions):
                for coli in range(n_col_regions):
                    row_start, row_end = rowi * krows_size, (rowi+1) * krows_size
                    row_end = min(row_end, n_rows)

                    col_start, col_end = coli * kcols_size, (coli+1) * kcols_size
                    col_end = min(col_end, n_cols)

                    grid_output[rowi][coli] = sim_matrix[row_start:row_end, col_start, col_end].min()

            grid_outputs.append(grid_output.view(-1))

        return torch.stack(grid_output, dim=0)
