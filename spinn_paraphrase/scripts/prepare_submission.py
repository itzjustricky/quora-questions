"""

    Take a csv file and prepare it to be submitted


"""


import argparse

import numpy as np
import pandas as pd


def main():

    parser = argparse.ArgumentParser(
        description='Fix a output file to be prepared for submission')

    parser.add_argument('-i', '--input_file')
    parser.add_argument('-t', '--test_size', type=int, default=2345796)
    parser.add_argument('-o', '--output_path')
    args = parser.parse_args()

    input_file = args.input_file

    output = pd.read_csv(input_file)
    output['id'] = output['id'].astype(np.int32)
    output = output.sort_values(by='id').reset_index(drop=True)

    all_ids = np.arange(args.test_size).astype(np.int32)
    output_ids = output['id']
    missing_ids = np.setdiff1d(all_ids, output_ids)
    n_missing = missing_ids.shape[0]

    output.columns = ['test_id', 'is_duplicate']

    # add in low probabilities for being duplicate for rows
    # containing sentences that couldn't be parsed
    preds_to_add = pd.DataFrame(
        np.array((missing_ids,
                  np.zeros(n_missing))).T,
        columns=output.columns)
    preds_to_add['test_id'] = preds_to_add['test_id'].astype(np.int32)

    assert((preds_to_add.shape[0] + output.shape[0]) == args.test_size)

    prepared_submission = pd.concat([output, preds_to_add])
    prepared_submission = prepared_submission.sort_values(by='test_id').reset_index(drop=True)

    assert(np.unique(prepared_submission['test_id'].values).shape[0] ==
           prepared_submission['test_id'].values.shape[0])
    prepared_submission.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()
