# Settings for the feature_extraction: Here you have to specify the event log attributes
FILE_NAME = 'CCC19 - Log XES' #Has to be stored in the input_event_logs directory
TRACE_ATTRIBUTE_NAME = 'case:concept:name'
ACTIVITY_ATTRIBUTE_NAME = 'concept:name'
TIMESTAMP_ATTRIBUTE_NAME = 'time:timestamp'
ORG_RESOURCE_ATTRIBUTE_NAME = 'org:resource'
ORG_ROLE_ATTRIBUTE_NAME = 'org:role'
CLASS_LABEL = 'task_type'
DATATYPE = 'XES'  # CSV, XES (default, CSV support is not fully guaranteed)
SEPARATOR = '' # CSV column separator
MODE = 'PREDICTION'  # PREPARATION (used to extract "automated" activities based on timestamp attribute), PREDICTION
TIMESTAMP_MODE = 'DEFAULT'  # DEFAULT (only time:timestamp), START_AND_AND (start timestamp and end
# timestamp available)

# List of attributes that are required to compute the features
ATTRIBUTE_LIST = [TRACE_ATTRIBUTE_NAME, ACTIVITY_ATTRIBUTE_NAME, TIMESTAMP_ATTRIBUTE_NAME, ORG_RESOURCE_ATTRIBUTE_NAME,
                  ORG_ROLE_ATTRIBUTE_NAME]

# Settings for bert_automation_indication
TEXT_COLUMNS = ['activity', 'business object', 'action']  # ['activity', 'business object', 'action']
BATCH_SIZE = 16

# Settings for classifier (predict.py)
# Do not change
LABEL_DICT = {'Automated': 0, 'High Automatable User Task': 2, 'Low Automatable User Task': 1,
              'Physical or Cognitive Task': 3}
# Do not change
LABEL_DICT_BO = {'Automated': 0, 'Physical or Cognitive Task': 1}

# Defines the features on which the classifier is trained (Currently, refers to feature set 9)
# If the classifier gets trained on different features, this list needs to get updated accordingly
FEATURE_SUBSET = ['C_activity_Automated',
                  'C_activity_Physical or Cognitive Task',
                  'C_activity_Low Automatable User Task',
                  'C_activity_High Automatable User Task',
                  'C_action_Automated',
                  'C_action_Low Automatable User Task',
                  'C_action_High Automatable User Task',
                  'C_action_Physical or Cognitive Task',
                  'C_business object_Automated',
                  'C_business object_Physical or Cognitive Task'
                  ]

# Defines the categorical features to be considered by one hot encoder
CAT_FEATURES = ['deterministic_following_activity',
                'deterministic_preceding_activity',
                'executing resource']

# Defines the numerical features that are transformed by the standard scaler
NUM_FEATURES = ['IT_relatedness', 'following_activities_standardization',
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

# The column order of the output
TARGET_ORDER = ['activity', 'Prob_High Automatable User Task', 'Prob_Low Automatable User Task',
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
