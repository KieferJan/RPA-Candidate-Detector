import pickle
import pandas as pd
import constants as c



def start(df):
   X, new_df = preprocess(df)
   predictions = predict(X)

   new_df = new_df.join(predictions)
   new_df = reorder(new_df)
   new_df = renameColumns(new_df)
   return new_df

def preprocess(df):
    # Adjust missing values
    values = {'activity': 'missing',
              'business object': 'missing',
              'action': 'missing',
              'executing resource': 'missing',
              'IT_relatedness': 0.0,
              'following_activities_standardization': 0.0,
              'preceding_activities_standardization': 0.0,
              'number_of_resources': 0.0,
              'ef_relative': 0.0,
              'median_execution_time': 0.0,
              'et_relative': 0.0,
              'C_activity_Automated': 0.0,
              'C_activity_Physical or Cognitive Task': 0.0,
              'C_activity_Low Automatable User Task': 0.0,
              'C_activity_High Automatable User Task': 0.0,
              'C_action_Automated': 0.0,
              'C_action_Physical or Cognitive Task': 0.0,
              'C_action_Low Automatable User Task': 0.0,
              'C_action_High Automatable User Task': 0.0,
              'C_business object_Automated': 0.0,
              'C_business object_Physical or Cognitive Task': 0.0}

    df.fillna(value=values, inplace=True)
    # df.dropna(inplace=True)
    na_free = df.dropna()
    na_free.reset_index(drop=True, inplace=True)
    only_na = df[~df.index.isin(na_free.index)]
    if not only_na.empty:
        print('Dropped rows due to missing values')
        print(only_na)

    numeric_features = ['IT_relatedness', 'following_activities_standardization',
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

    categorical_features = ['deterministic_following_activity',
                            'deterministic_preceding_activity']

    X = transform_data(na_free, numeric_features)

    return X, na_free


def transform_data(df, numeric_features):
    # numeric fetures scaling
    stscaler = pickle.load(open('./classifier/model/scaler.pkl', 'rb'))
    # onehotencoder = pickle.load(open('./classifier/model/onehotencoder.pkl', 'rb'))

    # transform numeric data
    X = pd.DataFrame(stscaler.transform(df[numeric_features]), columns=numeric_features)

    # transform categorical data

    X = X[c.FEATURE_SUBSET]

    return X

def predict(X):
    # and later you can load it
    with open('./classifier/model/svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)

    prediction = svm.predict_proba(X)

    result = pd.DataFrame({'Prob_Automated': prediction[:, 0], 'Prob_Low Automatable User Task': prediction[:, 1],
                           'Prob_High Automatable User Task': prediction[:, 2],
                           'Prob_Physical or Cognitive Task': prediction[:, 3]})

    return result

def reorder(df):
    target_order = ['activity', 'task_type', 'Prob_High Automatable User Task', 'Prob_Low Automatable User Task',
                    'Prob_Automated', 'Prob_Physical or Cognitive Task', 'IT_relatedness',
                    'deterministic_following_activity', 'deterministic_preceding_activity',
                    'following_activities_standardization', 'preceding_activities_standardization',
                    'failure_rate', 'number_of_resources', 'ef_relative', 'median_execution_time',
                    'et_relative', 'stability', 'business object', 'action', 'executing resource',
                    'C_activity_Automated', 'C_activity_Physical or Cognitive Task',
                    'C_activity_Low Automatable User Task',
                    'C_activity_High Automatable User Task',
                    'C_business object_Automated', 'C_business object_Physical or Cognitive Task',
                    'C_action_Automated', 'C_action_Low Automatable User Task',
                    'C_action_High Automatable User Task', 'C_action_Physical or Cognitive Task',
                    'process_name']
    df = df[target_order].sort_values(by='Prob_High Automatable User Task', ascending=False)

    return df

def renameColumns(df):
    df.rename(columns={'following_activities_standardization': 'standardization_f_e',
                       'preceding_activities_standardization': 'standardization_p_e',
                       'deterministic_following_activity': 'deterministic_f_e',
                       'deterministic_preceding_activity': 'deterministic_p_e',
                       'action': 'type_of_action',
                       'business object': 'type_of_business_object',
                       'executing resource': 'type_of_executing_resource',
                       'C_activity_Automated': 'C_eventlabel_Automated',
                       'C_activity_Physical or Cognitive Task': 'C_eventlabel_Physical or Cognitive Task',
                       'C_activity_Low Automatable User Task': 'C_eventlabel_Low Automatable User Task',
                       'C_activity_High Automatable User Task': 'C_eventlabel_High Automatable User Task',
                       'C_business object_Automated': 'C_businessobject_Digital',
                       'C_business object_Physical or Cognitive Task': 'C_businessobject_Physical'
                       })
    return df