"""
    Split the file randomly into two

"""

import os
import argparse

import numpy as np
# from pylore.tidy_tools import CSVDataSeparator


def read_in_jsons(filepath):
    sentences = []
    with open(filepath, 'r') as in_fh:
        for line in in_fh:
            sentences.append(line.strip())
    return sentences


def main():
    parser = argparse.ArgumentParser(
        description=' '.join((
            "Split a file's rows into validation and training sets",
            "and output them into new file"))
    )
    parser.add_argument('-i', '-input_path', dest='input_path', required=True, type=str,
                        help='The path to the input file to split')
    parser.add_argument('-t', '--train_fraction', dest='train_fraction', required=True, type=float,
                        help='The fraction of the data to set as training data')

    args = parser.parse_args()
    train_fraction = args.train_fraction
    input_path = args.input_path

    jsons_read = read_in_jsons(input_path)
    n_rows = len(jsons_read)
    all_rows = np.arange(n_rows)
    training_rows = np.random.choice(
        all_rows,
        size=int(n_rows * train_fraction),
        replace=False)

    output_dir = os.path.dirname(input_path)
    file_name, extension = os.path.splitext(os.path.basename(input_path))
    train_outpath = os.path.join(
        output_dir,
        '{}_train{}'.format(file_name, extension))
    dev_outpath = os.path.join(
        output_dir,
        '{}_dev{}'.format(file_name, extension))

    with open(train_outpath, 'w') as tout, open(dev_outpath, 'w') as dout:
        for ind, json_line in enumerate(jsons_read):
            # output to training file
            if ind in training_rows:
                tout.write(json_line)
                tout.write('\n')
            # output the dev file
            else:
                dout.write(json_line)
                dout.write('\n')

            if (ind % 1000) == 0:
                tout.flush(); dout.flush()


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
