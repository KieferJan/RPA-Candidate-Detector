import sys
import time
import bert_parser.preprocess.preprocessor as pp
import bert_parser.bert_tagger.bert_tagger as bt

# pre-processing the input data, i.e. splitting on underscores, camel case, etc.
PRE_PROCESS = True
# tag the log with semantic components using the BERT-based tagger (located in the ./model directory)
DO_BERT = PRE_PROCESS and True

# use if you want to load data from the project directory
DEFAULT_INPUT_DIR = 'bert_parser/input/'
# use if you want to write data to the project directory
DEFAULT_OUTPUT_DIR = 'bert_parser/output/'
# the location where the serialized model is stored
DEFAULT_MODEL_DIR = 'bert_parser/model/'

# dummy data for testing
test_data = ["create ORDER in the SAP_system",
             "create ORDER in the customer database",
             "User 123 forwards the bill to the manager and sends a confirmation",
             "select & send OFFER to the requesting customer0001",
             "check if order isReady or inProgress"]


def main(test_data):
    if PRE_PROCESS:
        tic = time.perf_counter()
        cleaned = [pp.preprocess_label(label) for label in test_data]
        toc = time.perf_counter()
        print(f"Preprocessed the data in {toc - tic:0.4f} seconds")
    if DO_BERT:
        print("BERT-based tagging")
        print("load model")
        tic = time.perf_counter()
        bert_tagger = bt.BertTagger()
        bert_tagger.load_trained_model(DEFAULT_MODEL_DIR)
        toc = time.perf_counter()
        print(f"Loaded the trained model in {toc - tic:0.4f} seconds")
        print('tagging text attributes')
        tagged = bert_tagger.get_tags_for_list(cleaned)
        tic = time.perf_counter()
        print(f"Tagged the whole data set in {tic - toc:0.4f} seconds")
        return tagged


if __name__ == '__main__':
    main_tic = time.perf_counter()
    main()
    main_toc = time.perf_counter()
    print(f"Program finished all operations in {main_toc - main_tic:0.4f} seconds")
    sys.exit()


# In[ ]:




