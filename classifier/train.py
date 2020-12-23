import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import constants as c

def start():
    X_train_trans, X_test_trans, y_train, y_test = preprocess()
    train(X_train_trans, X_test_trans, y_train, y_test)


def preprocess():
    # Adjust missing values
    X_train = pd.read_csv('./classifier/data/X_train.csv', index_col=0)
    X_test = pd.read_csv('./classifier/data/X_test.csv', index_col=0)
    y_train = pd.read_csv('./classifier/data/y_train.csv', index_col=0)
    y_test = pd.read_csv('./classifier/data/y_test.csv', index_col=0)

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

    X_train_trans, X_test_trans = transform_data(X_train, X_test, numeric_features, categorical_features)
    print('Finished PREPROCESS')
    return X_train_trans, X_test_trans, y_train, y_test


def transform_data(X_train, X_test, numeric_features, categorical_features):
    # numeric fetures scaling
    stscaler = StandardScaler()
    X_full = X_train.append(X_test)
    stscaler.fit(X_full[numeric_features])

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

    pickle.dump(stscaler, open('./classifier/model/scaler.pkl', 'wb'))
    pickle.dump(onehotencoder, open('./classifier/model/onehotencoder.pkl', 'wb'))

    return X_train_trans, X_test_trans


def train(X_train_trans, X_test_trans, y_train, y_test):
    print('Start Train')

    rf = RandomForestClassifier(random_state=42)

    model_desc = []
    model_rslt_training = []
    model_rslt_test = []
    cum_h_user_task = []
    cum_l_user_task = []
    acc_class1 = []
    acc_class2 = []

    f1 = cross_val_score(rf, X_train_trans[c.FEATURE_SUBSET], y_train.values.ravel(), cv=10,
                         scoring="f1_weighted")
    rf.fit(X_train_trans[c.FEATURE_SUBSET], y_train.values.ravel())
    prediction = rf.predict(X_test_trans[c.FEATURE_SUBSET])
    model_desc.append(f'Predictors perfomance')
    cm = confusion_matrix(y_test, prediction)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    model_rslt_training.append(f1.mean() * 100)
    model_rslt_test.append(calculate_measures(cm))
    cum_l_user_task.append(cm[1][1] + cm[1][2])
    cum_h_user_task.append(cm[2][1] + cm[2][2])
    acc_class1.append(cm[1][1])
    acc_class2.append(cm[2][2])


    print("Comparison of the models results on training an test")
    rslt = pd.DataFrame(pd.Series(model_desc, name="Model Description")).join(
    pd.DataFrame(pd.Series(model_rslt_training, name="F1 Training"))).join(pd.DataFrame(model_rslt_test))
    rslt['cum_acc_h_usertask'] = cum_h_user_task
    rslt['cum_acc_l_usertask'] = cum_l_user_task
    rslt['acc_class1'] = acc_class1
    rslt['acc_class2'] = acc_class2

    print(rslt)

    with open('./classifier/model/rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)


def calculate_measures(cm):
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    precision = cm[1][1] / cm.sum(axis=0)[1]
    recall = cm[1][1] / cm.sum(axis=1)[1]
    f1 = 2 * precision * recall / (precision + recall)

    return {"accuracy_test": accuracy, "precision_test": precision, "recall_test": recall, "f1_test": f1}
