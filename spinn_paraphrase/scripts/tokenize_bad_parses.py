"""
    Module to try to fix the parses that
    resulted in no data

"""

import argparse

import pandas as pd
from nltk import tokenize


def main():

    parser = argparse.ArgumentParser(
        description=' '.join((
            'Find bad parses and tokenize the original sentences and ',
            'output them into a new file indexed by their line #'))
    )
    parser.add_argument('--parsed_file', dest='parsed_file', required=True,
                        help='The path to the parse output of the Berkeley parser')
    parser.add_argument('--unparsed_file', dest='unparsed_file', required=True,
                        help='The path to the unparsed sentences fed into the Berkeley parser')
    parser.add_argument('--output_path', dest='output_path', required=True,
                        help='The path to the output of the file')
    args = parser.parse_args()

    parsed_file = args.parsed_file
    unparsed_file = args.unparsed_file
    output_path = args.output_path

    bad_parses = []
    parsed_sentences = pd.read_table(parsed_file, names=['sentence'], header=None)
    unparsed_sentences = pd.read_table(unparsed_file, names=['sentence'], header=None)
    assert(parsed_sentences.shape == unparsed_sentences.shape)

    bad_parse_indices = parsed_sentences.where(parsed_sentences.sentence == '(())').dropna().index
    for ind in bad_parse_indices:
        tokenized_string = tokenize.sent_tokenize(
            str(unparsed_sentences.iloc[ind].sentence))
        for sentence in tokenized_string:
            bad_parses.append((ind, sentence))

    with open(output_path, 'w') as out_fh:
        for tup in bad_parses:
            ind, sentence = tup
            out_fh.write('{},{}\n'.format(ind, sentence))


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
