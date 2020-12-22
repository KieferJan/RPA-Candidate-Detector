import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import constants as c



def start(df):
   X = preprocess(df)
   df = predict(X)

   return df

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
              'Confidence_activity_Automated': 0.0,
              'Confidence_activity_Physical or Cognitive Task': 0.0,
              'Confidence_action_Automated': 0.0,
              'Confidence_action_Physical or Cognitive Task': 0.0,
              'Confidence_action_Low Automatable User Task': 0.0,
              'Confidence_action_High Automatable User Task': 0.0,
              'Confidence_business object_Automated': 0.0,
              'Confidence_business object_Physical or Cognitive Task': 0.0}

    df.fillna(value=values, inplace=True)
    df.dropna(inplace=True)

    numeric_features = ['IT_relatedness', 'following_activities_standardization',
                        'preceding_activities_standardization', 'failure_rate',
                        'number_of_resources', 'ef_relative',
                        'median_execution_time', 'et_relative', 'stability',
                        'Confidence_activity_Automated',
                        'Confidence_activity_Physical or Cognitive Task',
                        'Confidence_action_Automated',
                        'Confidence_action_Physical or Cognitive Task',
                        'Confidence_action_Low Automatable User Task',
                        'Confidence_action_High Automatable User Task',
                        'Confidence_business object_Automated',
                        'Confidence_business object_Physical or Cognitive Task']

    categorical_features = ['deterministic_following_activity',
                            'deterministic_preceding_activity']

    X = transform_data(df[c.FEATURE_SUBSET], numeric_features, categorical_features)

    return X


def transform_data(X_train, numeric_features, categorical_features):
    # numeric fetures scaling
    stscaler = pickle.load(open('model/scaler.pkl','rb'))
    X_full = X_train.append(X_test)

    # transform numeric data
    X_train_trans = pd.DataFrame(stscaler.transform(X_train[numeric_features]), columns=numeric_features)
    X_test_trans = pd.DataFrame(stscaler.transform(X_test[numeric_features]), columns=numeric_features)

    # categorical features one hot encoding
    onehotencoder = OneHotEncoder()
    onehotencoder.fit(X_full[categorical_features])

    # transform categorical data
    X_train_trans = X_train_trans.join(
        pd.DataFrame(onehotencoder.transform(X_train[categorical_features]).toarray(),
                     columns=onehotencoder.get_feature_names(categorical_features)))
    X_test_trans = X_test_trans.join(pd.DataFrame(onehotencoder.transform(X_test[categorical_features]).toarray(),
                                                  columns=onehotencoder.get_feature_names(categorical_features)))

    return X_train_trans, X_test_trans

def predict(df):

    return df