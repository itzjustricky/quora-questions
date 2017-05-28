"""
    Model for Paraphrase detection

"""

import torch
import torch.nn as nn
from torch.autograd import Variable

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
        sentence1 = self.encoder(sentence1_embed, sentence1_trans)
        sentence2 = self.encoder(sentence2_embed, sentence2_trans)

        # TODO
        # replace with SimilarityMatrix here

        # pooling operation here

        # finally use softmax as before

        scores = self.out(
            torch.cat(
                [self.feature(sentence1, sentence2),
                 sentence_features(batch.sentence1, batch.sentence2, self.sentence_vocab)],
                1   # the dimensions to concatenate over
            )
        )

        return scores


class SimilarityMatrix(nn.Module):

    def __init__(self):
        super(SimilarityMatrix, self).__init__()

    def forward(self, states_1, states_2):
        pass
