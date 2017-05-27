"""
    Format the data to be readily read from the torchtext package

"""

# import os
import re
import json
import argparse
from copy import deepcopy

import pandas as pd

from nltk.tree import Tree
from nltk import tokenize

INDEX_PATTERN = re.compile('at index ([0-9]+).')
TREE_RESIDUE_PATTERN = re.compile('\|\s+<[\s-]+>')


def format_sentence_parse(parsed_string):
    """ Format the sentence parse to be binarized and strip the pos tags """
    try:
        return create_tree_string(parsed_string)
    # sometimes Berkeley parser outputs the wrong number of parenthesis
    except ValueError as err:
        print("Experienced Error:")
        print(err)
        print("With parsed_string: {}".format(parsed_string))
        error_string = str(err)

        # too little right brackets
        if "expected ')'" in error_string:
            # count number of left, right brackets
            n_left_brackets, n_right_brackets = parsed_string.count('('), parsed_string.count(')')
            n_difference = n_left_brackets - n_right_brackets
            parsed_string += n_difference * ')'
        # too many right brackets
        if "expected 'end-of-string'" in error_string:
            index = int(INDEX_PATTERN.findall(error_string)[0])
            parsed_string = parsed_string[:index]

        return create_tree_string(parsed_string)


def create_tree_string(parsed_string):
    parse_tree = Tree.fromstring(
        parsed_string,
        read_node=ignore_node_label,
        remove_empty_top_bracketing=False)
    parse_tree.chomsky_normal_form()
    parse_tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar='')
    return format_tree_string(str(parse_tree))


def create_json_dict(keys, content):
    json_dict = dict()
    for key, content in zip(keys, content):
        json_dict[key] = str(content)
    return json_dict


def ignore_node_label(node):
    """ Used to ignore POS tags """
    return ''


def format_tree_string(parse_tree_string):
    parse_tree_string = parse_tree_string.replace('\n', '')     # delete all newline
    parse_tree_string = ' '.join(parse_tree_string.split())     # replace many white space with 1
    # delete parenthesis surrounding single word
    parse_tokens = tokenize.word_tokenize(parse_tree_string)
    parse_tokens = unsurround_leaf_tokens(parse_tokens)

    sentence_parse = ' '.join(parse_tokens)
    return TREE_RESIDUE_PATTERN.sub('', sentence_parse)


def unsurround_leaf_tokens(tokens):
    """ Delete surrounding parenthesis for a leaf node """
    tokens = deepcopy(tokens)
    tokens_to_delete = []       # indices of tokens to delete
    for ind, token in enumerate(tokens):
        if token == ')':
            two_inds_back = max(ind - 2, 0)
            prev_tokens = tokens[two_inds_back:ind]
            if len(prev_tokens) == 2:
                if prev_tokens[0] == '(':
                    tokens_to_delete.append(two_inds_back)
                    tokens_to_delete.append(ind)

    for ind in sorted(tokens_to_delete, reverse=True):
        tokens.pop(ind)
    return tokens


def read_unparsed_sentences(filepath):
    sentences = []
    with open(filepath, 'r') as in_fh:
        for line in in_fh:
            sentences.append(line.strip())
    return sentences


def main():
    parser = argparse.ArgumentParser(
        description=' '.join((
            'Join the unparsed and parsed sentences',
            'together into a json document'))
    )

    parser.add_argument('-u1', '--unparsed_sentence1_file', dest='unparsed_sentence1_file',
                        required=True, default=None,
                        help='The path to the unparsed sentence1')
    parser.add_argument('-u2', '--unparsed_sentence2_file', dest='unparsed_sentence2_file',
                        required=True, default=None,
                        help='The path to the unparsed sentence2')

    parser.add_argument('-p1', '--parsed_sentence1_file', dest='parsed_sentence1_file', required=False,
                        help='The path to the parsed output of the Berkeley parser for sentence1')
    parser.add_argument('-p2', '--parsed_sentence2_file', dest='parsed_sentence2_file', required=False,
                        help='The path to the parsed output of the Berkeley parser for sentence2')

    parser.add_argument('-t', '--target_path', dest='target_path', required=False, default=None,
                        help='The path to the targets')

    parser.add_argument('-k', '--keep-index', dest='keep_index', action='store_true',
                        help='Indicates whether or not to keep the index values')
    parser.add_argument('-s', '--starting-index', dest='start_index', required=False, default=0, type=int,
                        help='The index to start at for outputting data')

    parser.add_argument('-o', '--output_path', dest='output_path', required=True,
                        help='The path to the output of the file')
    args = parser.parse_args()

    data_df = pd.DataFrame()
    # I have to read these explicitly line by line ...
    data_df['sentence1'] = read_unparsed_sentences(args.unparsed_sentence1_file)
    data_df['sentence2'] = read_unparsed_sentences(args.unparsed_sentence2_file)
    # optional data columns
    if args.parsed_sentence1_file is not None:
        data_df['sentence1_binary_parse'] = pd.read_table(
            args.parsed_sentence1_file, header=None).values
    if args.parsed_sentence2_file is not None:
        data_df['sentence2_binary_parse'] = pd.read_table(
            args.parsed_sentence2_file, header=None).values
    if args.target_path is not None:
        data_df['label'] = pd.read_table(args.target_path, header=None)
    output_path = args.output_path

    if data_df.isnull().values.any():
        raise ValueError("There are nan values in the resulting DataFrame, "
                         "probably because of mismatched # of rows.")

    start_index = args.start_index
    features = list(data_df.columns)
    with open(output_path, 'w') as out_fh:

        for ind, row_tuple in enumerate(data_df.iloc[start_index:].itertuples()):
            row_tuple = row_tuple[1:]

            json_dict = create_json_dict(features, row_tuple)
            if 'sentence1_binary_parse' in json_dict:
                if json_dict['sentence1_binary_parse'] == '(())':
                    continue
                else:
                    json_dict['sentence1_binary_parse'] = \
                        format_sentence_parse(json_dict['sentence1_binary_parse'])
            if 'sentence2_binary_parse' in json_dict:
                if json_dict['sentence1_binary_parse'] == '(())':
                    continue
                else:
                    json_dict['sentence2_binary_parse'] = \
                        format_sentence_parse(json_dict['sentence2_binary_parse'])

            if args.keep_index:
                json_dict['id'] = str(ind)

            json.dump(json_dict, out_fh, sort_keys=True)
            out_fh.write('\n')


if __name__ == '__main__':
    main()
"""


# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
"""
