# constans for the feature_extraction

FILE_NAME = 'CreditRequirement'
TRACE_ATTRIBUTE_NAME = 'case:concept:name'
ACTIVITY_ATTRIBUTE_NAME = 'concept:name'
TIMESTAMP_ATTRIBUTE_NAME = 'time:timestamp'
ORG_RESOURCE_ATTRIBUTE_NAME = 'org:resource'
ORG_ROLE_ATTRIBUTE_NAME = 'org:role'
CLASS_LABEL = 'task_type'
DATATYPE = 'XES' # CSV, XES
SEPARATOR = ''
MODE = 'TRAINING' # PREPARATION, TRAINING, PREDICTION
TIMESTAMP_MODE = 'START_AND_END' # DEFAULT, START_AND_AND

ATTRIBUTE_LIST = [TRACE_ATTRIBUTE_NAME, ACTIVITY_ATTRIBUTE_NAME, TIMESTAMP_ATTRIBUTE_NAME, ORG_RESOURCE_ATTRIBUTE_NAME, ORG_ROLE_ATTRIBUTE_NAME]

# constants for activity_bert_parser
DO_TRAIN_BERT = False
DO_PREDICT_BERT = True
TEXT_COLUMNS = ['action'] # ['activity', 'business object', 'action']

# constants for classifier
