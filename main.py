import pandas as pd
from pm4py.util import constants

import feature_extraction as fe
import bert_automation_indication.activity_bert_parser as AutomationIndication
import classifier.train as train
import classifier.predict as predict

# Settings can be defined in here
import constants as c


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Workflow Step
    # default_df = fe.import_data()
    # distinct_activity_features_df = fe.extract_activity_features(default_df)
    # full_activity_features_df = fe.extract_activity_features_full_log(default_df)

    # join the two data frames (full trace into unique activities)
    # activity_df = fe.join_full_in_distinct(full_activity_features_df, distinct_activity_features_df)

    # log purpose
    # print(activity_df)
    # print(full_activity_features_df['activity'].unique())

    #Workflow Step
    # activity_df.to_csv(r'/Users/jankiefer/Documents/Studium/Master/Semester/5. Semester/RPA detector/output/{}.csv'.format(c.FILE_NAME), index=False, header=True)
    # full_activity_features_df.to_csv(r'/Users/jankiefer/Documents/Studium/Master/Semester/5. Semester/RPA detector/full.csv', index=False, header=True)

    df = pd.read_csv('./input.csv')
    # df = AutomationIndication.apply_bert(df)

    if c.DO_PREDICT_CLASSIFIER:
        print(df)
    else:
        print('Start train')
        train.start()
        #df = predictor(df)
    # print(df)
    df.to_csv('./result.csv', index=False, header=True)


