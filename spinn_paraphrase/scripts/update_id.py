"""

    This script is to update the id field for a jsonl file


"""

import json
import argparse


def main():

    id_adjustment = 1468726
    # json_file = 'quora_questions/'

    parser = argparse.ArgumentParser(
        description='Adjust the id for all the json documents in a jsonl file')
    parser.add_argument('-i', '--input_path', dest='input_path', required=True,
                        help='jsonl file to adjust the id for.')
    parser.add_argument('-o', '--output_path', dest='output_path', required=True,
                        help='The output path for the resulting new jsonl file')
    parsed_args = parser.parse_args()
    input_path = parsed_args.input_path
    output_path = parsed_args.output_path

    with open(input_path, 'r') as in_fh, open(output_path, 'w') as out_fh:
        for line in in_fh:
            dict_tmp = json.loads(line)
            dict_tmp['id'] = str(int(dict_tmp['id']) + id_adjustment)

            json.dump(dict_tmp, out_fh, sort_keys=True)
            out_fh.write('\n')


if __name__ == '__main__':
    main()
