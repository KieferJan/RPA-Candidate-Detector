# Settings for the feature_extraction
FILE_NAME = 'Sepsis Cases - Event Log'
TRACE_ATTRIBUTE_NAME = 'case:concept:name'
ACTIVITY_ATTRIBUTE_NAME = 'concept:name'
TIMESTAMP_ATTRIBUTE_NAME = 'time:timestamp'
ORG_RESOURCE_ATTRIBUTE_NAME = 'org:resource'
ORG_ROLE_ATTRIBUTE_NAME = 'org:role'
CLASS_LABEL = 'task_type'
DATATYPE = 'XES'  # CSV, XES
SEPARATOR = ''
MODE = 'TRAINING'  # PREPARATION, TRAINING, PREDICTION
TIMESTAMP_MODE = 'START_AND_END'  # DEFAULT, START_AND_AND

ATTRIBUTE_LIST = [TRACE_ATTRIBUTE_NAME, ACTIVITY_ATTRIBUTE_NAME, TIMESTAMP_ATTRIBUTE_NAME, ORG_RESOURCE_ATTRIBUTE_NAME,
                  ORG_ROLE_ATTRIBUTE_NAME]

# Settings for activity_bert_parser for
# For Training, use only one text column as the best model needs to be selected and stored
TEXT_COLUMNS = ['activity', 'business object', 'action']  # ['activity', 'business object', 'action']
BATCH_SIZE = 16

# Settings for classifier
LABEL_DICT_ACTION = {'Automated': 0, 'High Automatable User Task': 2, 'Low Automatable User Task': 1,
              'Physical or Cognitive Task': 3}
LABEL_DICT = {'Automated': 0, 'Physical or Cognitive Task': 1}
FEATURE_SUBSET = ['Confidence_action_Automated',
                  'Confidence_action_Low Automatable User Task',
                  'Confidence_action_High Automatable User Task',
                  'Confidence_action_Physical or Cognitive Task']
