import os
import sys
import time
import glob
from pprint import pprint

import torch
from torch import optim
from torch import nn

import data_loader
from data_loader import QuoraDataset
from torchtext import data

from fuzzy_dict import FuzzyDict

from model import ParaphraseClassifier
from util import get_args


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

# I need to change the source of data fed into here
answers = data.Field(sequential=False)
# train, dev, test = QuoraDataset.splits(inputs, answers, transitions, root='./quora_questions')
train, dev = QuoraDataset.splits(inputs, answers, transitions, root='./quora_questions')

# another filtering step because you know ... dirty data
bad_dev_data = get_bad_indices(dev.examples)
print("There are {} instances of bad validation data".format(len(bad_dev_data)))
for ind in sorted(bad_dev_data, reverse=True):
    dev.examples.pop(ind)

bad_train_data = get_bad_indices(train.examples)
print("There are {} instances of bad training data".format(len(bad_train_data)))
for ind in sorted(bad_train_data, reverse=True):
    train.examples.pop(ind)
sys.stdout.flush()

# inputs.build_vocab(train, dev)
inputs.build_vocab(train, dev, stoi_class=FuzzyDict)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)
    else:
        inputs.vocab.load_vectors(
            wv_dir=args.data_cache,
            wv_type=args.word_vectors,
            # wv_dim=args.d_embed, unk_init='zero')
            wv_dim=args.d_embed)

        os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
        torch.save(inputs.vocab.vectors, args.vector_cache)

answers.build_vocab(train)

# see what the data format at this point is
# I think this groups sentences that have the same length together

# train_iter, dev_iter, test_iter = data.BucketIterator.splits(
#     (train, dev, test), batch_size=args.batch_size, device=args.gpu)
train_iter, dev_iter = data.BucketIterator.splits(
    (train, dev), batch_size=args.batch_size, device=args.gpu)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers
if config.birnn:
    config.n_cells *= 2

if config.spinn:
    # config.lr = 2e-3    # 3e-4
    config.lr = 3e-4
    config.lr_decay_by = 0.75
    config.lr_decay_every = 1   # 0.6
    config.regularization = 0   # 3e-6
    config.mlp_dropout = 0.07
    config.embed_dropout = 0.08     # 0.17
    config.n_mlp_layers = 2
    config.d_tracker = 64
    config.d_mlp = 1024
    config.d_hidden = 300
    config.d_embed = 300
    config.d_proj = 600
    torch.backends.cudnn.enabled = False
else:
    config.regularization = 0

model = ParaphraseClassifier(config)
if config.spinn:
    model.out[len(model.out._modules) - 2].weight.data.uniform_(-0.005, 0.005)
if args.word_vectors:
    # this sets the embedding indices to the vocabulary
    # built above, could lead to index out-of-bounds if
    # using cached vocabulary
    model.embed.weight.data = inputs.vocab.vectors
if args.gpu != -1:
    model.cuda()
if args.resume_snapshot:
    model.load_state_dict(torch.load(args.resume_snapshot))

print("The args for this run are")
pprint(args.__dict__)
print('-' * 50)
print("The model is: ")
print(model)
print('-' * 50)

# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
# opt = optim.Adam(model.parameters(), lr=args.lr)
opt = optim.RMSprop(model.parameters(), lr=config.lr, alpha=0.9, eps=1e-6,
                    weight_decay=config.regularization)

iterations = 0
start = time.time()
best_dev_loss = 1.0
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
print(header)
sys.stdout.flush()

for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct = n_total = train_loss = 0

    for batch_idx, batch in enumerate(train_iter):
        model.train(); opt.zero_grad()
        for pg in opt.param_groups:
            pg['lr'] = args.lr * (args.lr_decay_by ** (
                iterations / len(train_iter) / args.lr_decay_every))
        iterations += 1
        answer = model(batch)
        # print(nn.functional.softmax(answer[0]).data.tolist(), batch.label.data[0])
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total
        loss = criterion(answer, batch.label)
        loss.backward(); opt.step(); train_loss += loss.data[0] * batch.batch_size

        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = \
                snapshot_prefix + \
                '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(
                    train_acc, train_loss / n_total, iterations)
            torch.save(model.state_dict(), snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        if iterations % args.dev_every == 0:
            model.eval(); dev_iter.init_epoch()
            n_dev_correct = dev_loss = 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                answer = model(dev_batch)

                n_dev_correct += (
                    torch.max(answer, 1)[1].view(dev_batch.label.size()).data ==
                    dev_batch.label.data).sum()
                dev_loss += criterion(answer, dev_batch.label).data[0] * dev_batch.batch_size
            dev_acc = 100. * n_dev_correct / len(dev)
            dev_loss = dev_loss / len(dev)
            print(dev_log_template.format(time.time()-start,
                  epoch, iterations, 1+batch_idx, len(train_iter),
                  100. * (1+batch_idx) / len(train_iter), train_loss / n_total, dev_loss,
                  train_acc, dev_acc))
            sys.stdout.flush()
            n_correct = n_total = train_loss = 0

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')

                snapshot_path = \
                    snapshot_prefix + \
                    '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(
                        dev_acc, dev_loss, iterations)
                torch.save(model.state_dict(), snapshot_path)

                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:
            print(log_template.format(time.time()-start,
                  epoch, iterations, 1+batch_idx, len(train_iter),
                  100. * (1+batch_idx) / len(train_iter), train_loss / n_total,
                  ' '*8, n_correct / n_total*100, ' '*12))
            sys.stdout.flush()
            n_correct = n_total = train_loss = 0
