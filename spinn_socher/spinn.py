import bpdb
"""
    James Bradbury's SPINN code

    Math-heavy, GPU-accelerated operations that benefit from
    batched execution take place in Tracker and Reduce

"""


import torch
from torch import nn
from torch.autograd import Variable
# from torch.nn import functional as F

import itertools
from collections import defaultdict


def tree_lstm(c1, c2, lstm_in):
    a, i, f1, f2, o = lstm_in.chunk(5, 1)
    c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
    h = o.sigmoid() * c.tanh()
    return h, c


def bundle(lstm_iter):
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    # why return two chunks?
    return torch.cat(lstm_iter, 0).chunk(2, dim=1)


def unbundle(state):
    if state is None:
        return itertools.repeat(None)
    return torch.split(torch.cat(state, 1), 1, 0)


def bundle_inner_nodes(node_maps):
    # batch_size = len(node_maps)

    bundles = []
    for node_map in node_maps:
        bundled_nodes = []
        node_depths = sorted(list(node_map.keys()))

        for i in range(node_depths):
            bundled_nodes.extend(node_map[i])
        bundles.append(torch.cat(bundled_nodes, 0))

    return bundles


class Reduce(nn.Module):
    """ TreeLSTM composition module for SPINN """

    def __init__(self, size, tracker_size=None):
        super(Reduce, self).__init__()
        # needs to output state for c & h of left and right as well as tracker state?
        # only need one bias term since invovlves sum of left and right and bias
        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)

        if tracker_size is not None:
            self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def forward(self, left_in, right_in, tracking=None):
        """Perform batched TreeLSTM composition.

        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCED.

        The TreeLSTM has two or three inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model. All are provided as
        iterables and batched internally into tensors.

        Additionally augments each new node with pointers to its children.

        Args:
            left_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the left child of each node
                in the batch.
            right_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the right child of each node
                in the batch.
            tracking: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the tracker LSTM state of
                each node in the batch, or None.

        Returns:
            out: Tuple of ``B`` ~autograd.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node. These
                objects are also augmented with ``left`` and ``right``
                attributes.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        lstm_in = self.left(left[0])
        lstm_in += self.right(right[0])
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking[0])
        out = unbundle(tree_lstm(left[1], right[1], lstm_in))
        # for o, l, r in zip(out, left_in, right_in):
        #     o.left, o.right = l, r
        return out


class Tracker(nn.Module):
    """ Keep track of sentence state updates the hidden state every time
        the network reads a word or applies Reduce.

        The Tracker produces a new state at every step of the stack manipulation
        (i.e., after reading each word or close parenthesis) given the current sentence
        context state, the top entry b in the buffer, and the top two entries s1, s2 in the stack:

        context[t+1] = Tracker(context[t], b, s1, s2)
    """

    def __init__(self, size, tracker_size, predict):
        """
        :param tracker_size: the # of hidden units to give the Tracker unit
        """
        super(Tracker, self).__init__()
        # input is the bundled tensors: b, s1, s2 where
        # b is top entry in buffer
        # s1, s2 are the top two entries in the stack
        self.rnn = nn.LSTMCell(3 * size, tracker_size)

        # I want to train transitions predictor to the passed parse input
        # it is not currently doing that but there seems to be some attempt
        if predict:
            # does this try to infer transition? i.e. create shift-reduce parse
            # why 4 outputs below?
            self.transition = nn.Linear(tracker_size, 4)
        self.state_size = tracker_size

    def reset_state(self):
        self.state = None

    def forward(self, bufs, stacks):
        buf = bundle(buf[-1] for buf in bufs)[0]
        # two stacks to get the top two elements per sample in batch
        stack1 = bundle(stack[-1] for stack in stacks)[0]
        stack2 = bundle(stack[-2] for stack in stacks)[0]
        x = torch.cat((buf, stack1, stack2), 1)

        if self.state is None:
            self.state = 2 * [Variable(
                x.data.new(x.size(0), self.state_size).zero_())]

        self.state = self.rnn(x, self.state)

        # predict the transitions also
        if hasattr(self, 'transition'):
            return unbundle(self.state), self.transition(self.state[0])

        return unbundle(self.state), None


class SPINN(nn.Module):

    def __init__(self, config):
        super(SPINN, self).__init__()
        self.config = config
        assert config.d_hidden == config.d_proj / 2
        self.reduce = Reduce(config.d_hidden, config.d_tracker)
        if config.d_tracker is not None:
            self.tracker = Tracker(config.d_hidden, config.d_tracker,
                                   predict=config.predict)

    def forward(self, buffers, transitions):
        """ Sentences of the same length are batched together and sent into this function

        :param buffers: vector representation of the sentences, words are represented
            by indices to word embedding
        :param transitions: indicates the parsing of the sentence structure;
            2, 3 represents shift, reduce respectively
        """

        batch_size = buffers.size(0)
        # The input comes in as a single tensor of word embeddings;
        # I need it to be a list of stacks, one for each example in
        # the batch, that we can pop from independently. The words in
        # each example have already been reversed, so that they can
        # be read from left to right by popping from the end of each
        # list; they have also been prefixed with a null value.
        # list of the wordvec tensors
        buffers = [list(torch.split(b.squeeze(1), 1, 0))    # remove all inputs of size 1
                   for b in torch.split(buffers, 1, 1)]     # get all buffers stored

        # track the right-to-left order of each depth in tree
        node_maps = [defaultdict(list) for i in range(batch_size)]

        for i, b in enumerate(torch.split(buffers, 1, 1)):
            # I need to make sure this is as expected
            bpdb.set_trace()  # ------------------------------ Breakpoint ------------------------------ #
            node_maps[i][0].extend(
                list(torch.split(b.squeeze(1), 1, 0)))

        # we also need two null values at the bottom of each stack,
        # so we can copy from the nulls in the input; these nulls
        # are all needed so that the tracker can run even if the
        # buffer or stack is empty
        stacks = [[buf[0], buf[0]] for buf in buffers]
        depth_tracker = [0 for i in range(batch_size)]

        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        else:
            # if no tracker then transitions cannot be inferred
            assert transitions is not None

        if transitions is not None:
            n_transitions = transitions.size(0)
            # trans_loss, trans_acc = 0, 0
        else:   # infer the # of transitions
            # there are n_words amount of shifts (all words must be pushed onto stack)
            # there are n_words-1 amount of reduces (operation done between two tokens)
            # -3 instead of -2 because of extra null values
            n_transitions = len(buffers[0]) * 2 - 3

        for i in range(n_transitions):
            if transitions is not None:
                trans = transitions[i]
            if hasattr(self, 'tracker'):
                # update tracker state and retrieve it
                # trans_hyp is a prediction of the transition from Tracker
                tracker_states, trans_hyp = self.tracker(buffers, stacks)

                # this overwrites the passed transitions data
                if trans_hyp is not None:
                    trans = trans_hyp.max(1)[1]     # get mostly likely transition

                    # trans_preds = trans_hyp.max(1)[1]
                    # if transitions is not None:
                    #     trans_loss += F.cross_entropy(trans_hyp, trans)
                    #     trans_acc += (trans_preds.data == trans.data).mean()
                    # else:
                    #     trans = trans_preds
            else:
                tracker_states = itertools.repeat(None)

            lefts, rights, trackings = [], [], []
            batch = zip(trans.data, buffers, stacks, tracker_states)

            cnt = 0
            for transition, buf, stack, tracking in batch:
                if transition == 3:     # shift
                    stack.append(buf.pop())
                    depth_tracker[cnt] += 1
                elif transition == 2:   # reduce
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
                    trackings.append(tracking)
                    depth_tracker[cnt] -= 1
                cnt += 1

            if rights:  # if none of batches had reduce then can skip
                reduced = iter(self.reduce(lefts, rights, trackings))
                # zip only goes over up to the end of the smaller of the two iterables
                # will only go through the one transition for each batch
                cnt = 0
                for transition, stack in zip(trans.data, stacks):
                    if transition == 2:
                        reduced_state = next(reduced)

                        node_maps[cnt][depth_tracker[cnt]].insert(0, reduced_state.clone())
                        stack.append(reduced_state)
                    cnt += 1

        return bundle_inner_nodes(node_maps)

        # if trans_loss is not 0:
        # bundle all the states together from the batch
        # return bundle([stack.pop() for stack in stacks])[0]
