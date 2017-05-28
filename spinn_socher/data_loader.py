import os
# from nltk.tokenize import word_tokenize

from torchtext import data


class ShiftReduceField(data.Field):

    def __init__(self):

        super(ShiftReduceField, self).__init__(preprocessing=lambda parse: [
            'reduce' if t == ')' else 'shift' for t in parse if t != '('])

        self.build_vocab([['reduce'], ['shift']])


class ParsedTextField(data.Field):

    def __init__(self, eos_token='<pad>', lower=False):

        super(ParsedTextField, self).__init__(
            eos_token=eos_token, lower=lower,
            preprocessing=lambda parse: [
                t for t in parse if t not in ('(', ')')],
            postprocessing=lambda parse, _, __: [
                list(reversed(p)) for p in parse])
        # tokenize=parsed_text_tokenizer)


class NumberField(data.Field):

    def __init__(self, **kwargs):

        super(NumberField, self).__init__(
            use_vocab=True, sequential=False)


class QuoraDataset(data.TabularDataset):

    # url = 'http://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    # filename = 'snli_1.0.zip'
    # dirname = 'snli_1.0'
    file_prefix = 'quora_questions_'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.sentence1), len(ex.sentence2))

    @classmethod
    def splits(cls, text_field, label_field, parse_field=None, root='.',
               train='train.jsonl', validation='dev.jsonl', test=None):
        """Create dataset objects for splits of the SNLI dataset.

        This is the most flexible way to use the dataset.

        Arguments:
            text_field: The field that will be used for sentence1 and sentence2
                data.
            label_field: The field that will be used for label data.
            parse_field: The field that will be used for shift-reduce parser
                transitions, or None to not include them.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose snli_1.0
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.jsonl'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.jsonl'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.jsonl'.
        """

        if parse_field is None:
            return super(QuoraDataset, cls).splits(
                os.path.join(root, cls.file_prefix), train, validation, test=test,
                format='json', fields={'sentence1': ('sentence1', text_field),
                                       'sentence2': ('sentence2', text_field),
                                       'label': ('label', label_field)},
                filter_pred=lambda ex: ex.label != '-')
        return super(QuoraDataset, cls).splits(
            os.path.join(root, cls.file_prefix), train, validation, test,
            format='json',
            fields={
                'sentence1_binary_parse':
                    [('sentence1', text_field),
                     ('sentence1_transitions', parse_field)],
                'sentence2_binary_parse':
                    [('sentence2', text_field),
                     ('sentence2_transitions', parse_field)],
                'label': ('label', label_field)}
        )

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.', wv_dir='.',
              wv_type=None, wv_dim='300d', trees=False, **kwargs):
        """Create iterator objects for splits of the SNLI dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Arguments:
            batch_size: Batch size.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            trees: Whether to include shift-reduce parser transitions.
                Default: False.
            Remaining keyword arguments: Passed to the splits method.
        """
        if trees:
            TEXT = ParsedTextField()
            TRANSITIONS = ShiftReduceField()
        else:
            TEXT = data.Field(tokenize='spacy')
            TRANSITIONS = None
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(
            TEXT, LABEL, TRANSITIONS, root=root, **kwargs)

        TEXT.build_vocab(train, wv_dir=wv_dir, wv_type=wv_type, wv_dim=wv_dim)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)


class QuoraTestDataset(data.TabularDataset):

    file_prefix = 'quora_questions_'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.sentence1), len(ex.sentence2))

    @classmethod
    def splits(cls, text_field, label_field, parse_field=None, root='.',
               train=None, validation=None, test='test.jsonl'):
        """Create dataset objects for splits of the SNLI dataset.

        This is the most flexible way to use the dataset.

        Arguments:
            text_field: The field that will be used for sentence1 and sentence2
                data.
            label_field: The field that will be used for label data.
            parse_field: The field that will be used for shift-reduce parser
                transitions, or None to not include them.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose snli_1.0
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.jsonl'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.jsonl'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.jsonl'.
        """

        if parse_field is None:
            return super(QuoraTestDataset, cls).splits(
                os.path.join(root, cls.file_prefix),
                train, validation, test=test, format='json',
                fields={
                    'sentence1':
                        ('sentence1', text_field),
                    'sentence2':
                        ('sentence2', text_field),
                    'id': ('id', label_field),
                }
            )
        return super(QuoraTestDataset, cls).splits(
            os.path.join(root, cls.file_prefix), train, validation, test,
            format='json',
            fields={
                'sentence1_binary_parse':
                    [('sentence1', text_field),
                     ('sentence1_transitions', parse_field)],
                'sentence2_binary_parse':
                    [('sentence2', text_field),
                     ('sentence2_transitions', parse_field)],
                'id': ('id', label_field),
            }
        )

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.', wv_dir='.',
              wv_type=None, wv_dim='300d', trees=False, **kwargs):
        """Create iterator objects for splits of the SNLI dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Arguments:
            batch_size: Batch size.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            trees: Whether to include shift-reduce parser transitions.
                Default: False.
            Remaining keyword arguments: Passed to the splits method.
        """
        if trees:
            TEXT = ParsedTextField()
            TRANSITIONS = ShiftReduceField()
        else:
            TEXT = data.Field(tokenize='spacy')
            TRANSITIONS = None
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(
            TEXT, LABEL, TRANSITIONS, root=root, **kwargs)

        TEXT.build_vocab(train, wv_dir=wv_dir, wv_type=wv_type, wv_dim=wv_dim)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)
