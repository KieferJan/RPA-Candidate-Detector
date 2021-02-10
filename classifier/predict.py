import pickle
import pandas as pd
import constants as c

# Define Workflow
def start(df):
    X = preprocess(df)
    predictions = predict(X)

    df = df.join(predictions)
    df = reorder(df)
    df = renameColumns(df)
    return df

# If values are missing add default values
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

    X = transform_data(df)

    return X

# Apply standard scaler and one hot encoder and return the data frame with the feature subset on which the SVM is trained
def transform_data(df):
    # numeric fetures scaling
    stscaler = pickle.load(open('./classifier/model/scaler.pkl', 'rb'))
    onehotencoder = pickle.load(open('./classifier/model/onehotencoder.pkl', 'rb'))

    # transform numeric data
    X_trans = pd.DataFrame(stscaler.transform(df[c.NUM_FEATURES]), columns=c.NUM_FEATURES)

    # transform categorical data
    X_trans = X_trans.join(pd.DataFrame(onehotencoder.transform(df[c.CAT_FEATURES]).toarray(),
                                        columns=onehotencoder.get_feature_names(c.CAT_FEATURES)))

    X_trans = X_trans[c.FEATURE_SUBSET]

    return X_trans

# Load model and predict probability
def predict(X):
    # and later you can load it
    with open('./classifier/model/svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)

    prediction = svm.predict_proba(X)

    result = pd.DataFrame({'Prob_Automated': prediction[:, 0], 'Prob_Low Automatable User Task': prediction[:, 1],
                           'Prob_High Automatable User Task': prediction[:, 2],
                           'Prob_Physical or Cognitive Task': prediction[:, 3]})

    return result

# Apply order of the output file
def reorder(df):
    df = df[c.TARGET_ORDER].sort_values(by='Prob_High Automatable User Task', ascending=False)

    return df

# Rename columns to be in line with the feature names mentioned in the thesis
def renameColumns(df):
    df.rename(columns={'following_activities_standardization': 'standardization_f_e',
                       'preceding_activities_standardization': 'standardization_p_e',
                       'deterministic_following_activity': 'deterministic_f_e',
                       'deterministic_preceding_activity': 'deterministic_p_e',
                       'action': 'type_of_action',
                       'business object': 'type_of_business_object',
                       'executing resource': 'type_of_executing_resource',
                       'C_activity_Automated': 'event label is C_A',
                       'C_activity_Physical or Cognitive Task': 'event label is C_P',
                       'C_activity_Low Automatable User Task': 'event label is C_L',
                       'C_activity_High Automatable User Task': 'event label is C_H',
                       'C_business object_Automated': 'BO is Digital',
                       'C_business object_Physical or Cognitive Task': 'BO is Physical',
                       'C_action_Automated': 'action is C_A',
                       'C_action_Low Automatable User Task': 'action is C_L',
                       'C_action_High Automatable User Task': 'action is C_H',
                       'C_action_Physical or Cognitive Task': 'action is C_P',
                       })
    return df
