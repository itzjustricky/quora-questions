"""
    Script to compare answers to the model output/prediction

"""

import json

import numpy as np
import pandas as pd


def main():
    outputs_are_probabilities = True
    json_file_path = 'quora_questions/subset_test/quora_questions_dev.jsonl'
    model_output_path = 'results/test_output.csv'

    predicted_target = pd.read_csv(model_output_path)
    with open(json_file_path, 'r') as fh:
        real_target = pd.DataFrame(json.loads(line) for line in fh)

    real_target['id'] = real_target['id'].astype(np.int64)
    real_target['label'] = real_target['label'].astype(np.int64)

    left_over_ids = sorted(list(
        set.intersection(
            set(predicted_target['id'].values),
            set(real_target['id'].values))
    ))
    n_rows = len(left_over_ids)

    # filter them to only the matching ids
    # some are taken out because of bad rows for prediction
    predicted_target = predicted_target[predicted_target.id.isin(left_over_ids)].sort_values(by='id')
    real_target = real_target[real_target.id.isin(left_over_ids)].sort_values(by='id')

    if outputs_are_probabilities:
        predicted_target['label'] = predicted_target['label'].round()

    n_correct = np.sum(predicted_target['label'].values == real_target['label'].values)
    print('#' * 100)
    print("The number of correct responses: {} ({} %)".format(n_correct, n_correct / n_rows * 100))
    print('#' * 100)
    print()

    # predicted_target['label'] =
    the_wrong_indices = np.where(predicted_target['label'].values != real_target['label'].values)[0]
    sampled_wrong_indices = np.random.choice(the_wrong_indices, 10)

    print("Random samples of incorrect")
    print('#' * 100)
    for ind in sampled_wrong_indices:
        pred_row = predicted_target.iloc[ind]
        real_row = real_target.iloc[ind]
        assert(pred_row['id'] == real_row['id'])
        print("Was wrong for id {}".format(real_row['id']))
        print('-' * 30)
        print(real_row['sentence1'])
        print(real_row['sentence2'])
        print(real_row['sentence1_binary_parse'])
        print(real_row['sentence2_binary_parse'])
        print("Correct: {}".format(real_row['label']))
        print("Predicted: {}".format(pred_row['label']))
        print()

    the_right_indices = np.where(predicted_target['label'].values == real_target['label'].values)[0]
    sampled_right_indices = np.random.choice(the_right_indices, 10)

    print("Random samples of correct")
    print('#' * 100)
    for ind in sampled_right_indices:
        pred_row = predicted_target.iloc[ind]
        real_row = real_target.iloc[ind]
        assert(pred_row['id'] == real_row['id'])
        print("Was wrong for id {}".format(real_row['id']))
        print('-' * 30)
        print(real_row['sentence1'])
        print(real_row['sentence2'])
        print(real_row['sentence1_binary_parse'])
        print(real_row['sentence2_binary_parse'])
        print("Correct: {}".format(real_row['label']))
        print("Predicted: {}".format(pred_row['label']))
        print()


if __name__ == '__main__':
    main()
