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
LABEL_DICT = {'Automated': 0, 'High Automatable User Task': 2, 'Low Automatable User Task': 1,
              'Physical or Cognitive Task': 3}
LABEL_DICT_BO = {'Automated': 0, 'Physical or Cognitive Task': 1}
FEATURE_SUBSET =  ['C_activity_Automated',
                   'C_activity_Physical or Cognitive Task',
                   'C_activity_Low Automatable User Task',
                   'C_activity_High Automatable User Task',
                   'C_action_Automated',
                   'C_action_Low Automatable User Task',
                   'C_action_High Automatable User Task',
                   'C_action_Physical or Cognitive Task',
                   'C_business object_Automated',
                   'C_business object_Physical or Cognitive Task', 'IT_relatedness',
                   'deterministic_following_activity_False',
                   'deterministic_following_activity_True',
                   'deterministic_preceding_activity_False',
                   'deterministic_following_activity_True',
                   'following_activities_standardization',
                   'preceding_activities_standardization',
                   'failure_rate',
                   'ef_relative',
                   'median_execution_time',
                   'et_relative',
                   'stability',
                   'executing resource_administration',
                   'executing resource_budget owner',
                   'executing resource_competent authority',
                   'executing resource_director',
                   'executing resource_employee',
                   'executing resource_from customer',
                   'executing resource_missing',
                   'executing resource_owner',
                   'executing resource_party',
                   'executing resource_pre approver',
                   'executing resource_prefecture',
                   'executing resource_supervisor',
                   'executing resource_vendor']

ONE_HOT_COLS = ['executing resource_administration',
                       'executing resource_budget owner',
                       'executing resource_competent authority',
                       'executing resource_director',
                        'executing resource_employee',
                        'executing resource_from customer',
                        'executing resource_missing',
                        'executing resource_owner',
                        'executing resource_party',
                       'executing resource_pre approver',
                        'executing resource_prefecture',
                       'executing resource_supervisor',
                        'executing resource_vendor']

CAT_FEATURES = ['deterministic_following_activity',
                            'deterministic_preceding_activity',
                            'executing resource']
NUM_FEATURES =  ['IT_relatedness', 'following_activities_standardization',
                        'preceding_activities_standardization', 'failure_rate',
                        'number_of_resources', 'ef_relative',
                        'median_execution_time', 'et_relative', 'stability',
                        'C_activity_Automated',
                        'C_activity_Physical or Cognitive Task',
                        'C_activity_Low Automatable User Task',
                        'C_activity_High Automatable User Task',
                        'C_action_Automated',
                        'C_action_Physical or Cognitive Task',
                        'C_action_Low Automatable User Task',
                        'C_action_High Automatable User Task',
                        'C_business object_Automated',
                        'C_business object_Physical or Cognitive Task']

TARGET_ORDER = ['activity', 'task_type', 'Prob_High Automatable User Task', 'Prob_Low Automatable User Task',
                    'Prob_Automated', 'Prob_Physical or Cognitive Task',
                    'C_action_Automated', 'C_action_Low Automatable User Task',
                    'C_action_High Automatable User Task', 'C_action_Physical or Cognitive Task',
                    'C_activity_Automated', 'C_activity_Physical or Cognitive Task',
                    'C_activity_Low Automatable User Task',
                    'C_activity_High Automatable User Task',
                    'C_business object_Automated', 'C_business object_Physical or Cognitive Task',
                    'IT_relatedness',
                    'deterministic_following_activity', 'deterministic_preceding_activity',
                    'following_activities_standardization', 'preceding_activities_standardization',
                    'failure_rate', 'number_of_resources', 'ef_relative', 'median_execution_time',
                    'et_relative', 'stability', 'business object', 'action', 'executing resource',
                    'process_name']