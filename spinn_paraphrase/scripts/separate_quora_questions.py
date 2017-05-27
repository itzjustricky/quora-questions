"""
    Separate the quora questions into

"""

import nltk.tokenize
import pandas as pd


def dump_sentences_from_dataframe(sentences_series, output_path):
    gathered_sentences = []
    for ind, sentence in sentences_series.iteritems():
        gathered_sentences.append(sentence)
    with open(output_path, 'w') as out_fh:
        out_fh.write('\n'.join(gathered_sentences))


def extract_and_output(datafile_path, output_paths, columns_to_extract,
                       map_fns=None, pd_kwargs=None):
    """ Extract the columns of csv data in the passed file and output
        the data each in their own file

    :param datafile_path: path to the csv data file
    :param output_paths: if this is a str then it will be used as a prefix,
            followed by _[xx] where xx is the index of the column in the file
    :param columns_to_extract: if None, all columns will be outputted
    """
    if pd_kwargs is None:
        pd_kwargs = dict()
    if map_fns is None:
        map_fns = dict()

    data_df = pd.read_csv(datafile_path, **pd_kwargs)[columns_to_extract]
    for out_path, col in zip(output_paths, columns_to_extract):
        if col in map_fns:
            data_series_tmp = data_df[col] \
                .reset_index(drop=True) \
                .apply(map_fns[col])
            dump_sentences_from_dataframe(data_series_tmp, out_path)
        else:
            data_series_tmp = data_df[col] \
                .reset_index(drop=True)
            data_series_tmp.to_csv(out_path, index=False)


def format_sentence(sentence_string):
    sentence_string = str(sentence_string).replace('\n', '')
    sentence_string = ' '.join(nltk.tokenize.word_tokenize(sentence_string))

    return sentence_string


def main():
    csv_data_path = './quora_questions/test_set/test.csv'
    # csv_data_path = './quora_questions/test.csv'
    output_paths = [
        './quora_questions/test_set/sentence1.txt',
        './quora_questions/test_set/sentence2.txt']
    # './quora_questions/target.txt']
    # columns_to_extract = ['question1', 'question2', 'is_duplicate']
    columns_to_extract = ['question1', 'question2']
    map_fns = {
        'question1': format_sentence,
        'question2': format_sentence, }

    extract_and_output(csv_data_path, output_paths, columns_to_extract, map_fns)


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
