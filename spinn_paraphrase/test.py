import os
import sys

import numpy as np
import pandas as pd

import torch
# from torch import nn
from torchtext import data

import data_loader
from data_loader import QuoraTestDataset
from fuzzy_dict import FuzzyDict

from model import ParaphraseClassifier
from model import ParaphraseClassifierExtra
from util import get_args

output_every_n_batches = 100
output_probabilities = True
output_file = 'output_spinn_2.csv'


def get_bad_indices(examples_list):
    bad_inds = []
    for ind, example in enumerate(examples_list):
        try:
            assert((2 * len(example.sentence1) - 1) == len(example.sentence1_transitions))
            assert((2 * len(example.sentence2) - 1) == len(example.sentence2_transitions))
            assert((example.sentence1_transitions.count("shift") - 1) == example.sentence1_transitions.count("reduce"))
            assert((example.sentence2_transitions.count("shift") - 1) == example.sentence2_transitions.count("reduce"))
        except AssertionError:
            bad_inds.append(ind)
    return bad_inds


def output_data_so_far(indices, outputs, output_file, append_to_file=True):

    # test_output_file = 'test_output.pkl'
    catted_indices = np.concatenate(indices)
    # translate indices, annoying ... this seems to be the only way
    # to get what I want. Setting use_vocab=False gives an error
    catted_indices_copy = catted_indices.copy()
    for i, ind in enumerate(catted_indices_copy):
        catted_indices[i] = index_field.vocab.itos[ind]
    catted_indices = catted_indices.astype(np.int32)
    catted_answers = np.concatenate(outputs)
    if not output_probabilities:
        catted_answers -= 1

    output_answers = pd.DataFrame(np.array([catted_indices, catted_answers]).T,
                                  columns=['id', 'label'])
    output_answers = output_answers.sort_values(by='id').reset_index(drop=True)
    if append_to_file:
        output_answers.to_csv(
            os.path.join(args.save_path, output_file),
            index=False, mode='a', header=False)
    else:
        output_answers.to_csv(
            os.path.join(args.save_path, output_file),
            index=False)


args = get_args()
if args.gpu != -1:
    print("Using the GPU")
    sys.stdout.flush()
    torch.cuda.set_device(args.gpu)

# something to note here is that ParsedTextField
# does an extra tokenize on the words in the sentence
if args.spinn:
    inputs = data_loader.ParsedTextField(lower=args.lower)
    transitions = data_loader.ShiftReduceField()
else:
    # inputs = data.Field(lower=args.lower)
    inputs = data_loader.ParsedTextField(lower=args.lower)
    transitions = None

# TODO: there is something to do here ...
# the choice of the Field object manipulates the indices that I have stored
# index_field = data_loader.NumberField(sequential=True)
# index_field = data.Field(sequential=False)
index_field = data_loader.NumberField()
# train, dev, test = QuoraDataset.splits(inputs, answers, transitions, root='./quora_questions')
test = QuoraTestDataset.splits(
    inputs, index_field, transitions,
    root='./quora_questions/test_set',
    # test='test.jsonl')[-1]
    # root='./quora_questions/subset_test',
    test='test.jsonl')[-1]

# another filtering step because you know ... dirty data
if args.spinn:
    bad_test_data = get_bad_indices(test.examples)
    bad_ids = []
    for ind in sorted(bad_test_data, reverse=True):
        bad_ids.append(test.examples.pop(ind).id)

    print("There are {} instances of bad testing data".format(len(bad_test_data)))
    print("They are {}".format(','.join(map(str, bad_ids))))
    sys.stdout.flush()
    vector_cache = os.path.join(os.getcwd(), '.vector_cache/test_vectors.pt')
else:
    vector_cache = os.path.join(os.getcwd(), '.vector_cache/test_vectors_plain.pt')

inputs.build_vocab(test, stoi_class=FuzzyDict)
# vector_cache = os.path.join(os.getcwd(), '.vector_cache/test_subset_vectors.pt')
if args.word_vectors:
    if os.path.isfile(vector_cache):
        inputs.vocab.vectors = torch.load(vector_cache)
    else:
        inputs.vocab.load_vectors(
            wv_dir=args.data_cache,
            wv_type=args.word_vectors,
            # wv_dim=args.d_embed, unk_init='zero')
            wv_dim=args.d_embed)

    os.makedirs(os.path.dirname(vector_cache), exist_ok=True)
    torch.save(inputs.vocab.vectors, vector_cache)

index_field.build_vocab(test)

args.batch_size = 400
# test_iter = data.Iterator.splits(
#     (test, ), batch_size=args.batch_size, device=args.gpu, repeat=False)[0]
test_iter = data.Iterator(
    test, batch_size=args.batch_size, device=args.gpu,
    repeat=False, train=False)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = 3
config.n_cells = config.n_layers
if config.birnn:
    config.n_cells *= 2

# if config.spinn:
#     config.lr = 2e-3    # 3e-4
#     config.lr_decay_by = 0.75
#     config.lr_decay_every = 1   # 0.6
#     config.regularization = 0   # 3e-6
#     config.mlp_dropout = 0.07
#     config.embed_dropout = 0.08     # 0.17
#     config.n_mlp_layers = 2
#     config.d_tracker = 64
#     config.d_mlp = 1024
#     config.d_hidden = 300
#     config.d_embed = 300
#     config.d_proj = 600
#     torch.backends.cudnn.enabled = False
# else:
#     config.regularization = 0

if config.spinn:
    config.lr = .001    # 3e-4
    config.lr_decay_by = 1
    config.lr_decay_every = 1   # 0.6
    config.regularization = 0   # 3e-6
    config.mlp_dropout = 0.5
    config.embed_dropout = 0.5     # 0.17
    config.n_mlp_layers = 3
    config.d_tracker = 200
    config.d_mlp = 600
    config.d_hidden = 300
    config.d_embed = 300
    config.d_proj = 600
    torch.backends.cudnn.enabled = False
else:
    config.regularization = 0


torch.backends.cudnn.benchmark = True

# model = ParaphraseClassifier(config)
model = ParaphraseClassifierExtra(config, inputs.vocab)
if config.spinn:
    model.out[len(model.out._modules) - 2].weight.data.uniform_(-0.005, 0.005)
if args.word_vectors:
    model.embed.weight.data = inputs.vocab.vectors
if args.gpu != -1:
    model.cuda()

# here I have to pass in resume_snapshot
if args.resume_snapshot:
    models_saved_state = torch.load(args.resume_snapshot)
    models_saved_state['embed.weight'] = inputs.vocab.vectors
    model.load_state_dict(models_saved_state)

# criterion = nn.CrossEntropyLoss()
sys.stdout.flush()

# ready the model for evaluation
the_test_outputs = []
the_test_indices = []
model.eval(); test_iter.init_epoch()
label_size = 2

n_times_outputted = 0

for test_batch_idx, test_batch in enumerate(test_iter):
    # if test_batch

    answer = model(test_batch)
    batch_size = test_batch.sentence1.data[1].size()    # torch.Size object

    if output_probabilities:
        # FOR PROBABILITY OUTPUTS
        model_output = torch.exp(answer)[:, 2].data.cpu().numpy()
        model_output = np.clip(model_output, 0.0, 1.0)
    else:
        # FOR LABEL OUTPUTS
        best_answers = torch.max(answer, 1)[1].view(batch_size).data
        model_output = best_answers.cpu().numpy()

    # view(test_batch.label.size()).data
    the_test_indices.append(test_batch.id.data.cpu().numpy())
    the_test_outputs.append(model_output)

    if (test_batch_idx % output_every_n_batches) == 0:
        print("[{}] Outputted data for batch {}".format(n_times_outputted, test_batch_idx))
        sys.stdout.flush()
        if n_times_outputted == 0:
            output_data_so_far(
                the_test_indices, the_test_outputs, output_file, append_to_file=False)
        else:
            output_data_so_far(
                the_test_indices, the_test_outputs, output_file, append_to_file=True)

        # clean up the outputted indices and outputs
        the_test_indices = []
        the_test_outputs = []
        n_times_outputted += 1


output_data_so_far(
    the_test_indices, the_test_outputs, output_file, append_to_file=True)
