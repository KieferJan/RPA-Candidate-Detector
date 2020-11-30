import pandas as pd

import feature_extraction as fe
import constants as c
from pm4py.util import constants
import bert_automation_indication.activity_bert_parser as AutomationIndication


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # default_df = fe.import_data()
    # distinct_activity_features_df = fe.extract_activity_features(default_df)
    # full_activity_features_df = fe.extract_activity_features_full_log(default_df)

    # join the two data frames (full trace into unique activities)
    # activity_df = fe.join_full_in_distinct(full_activity_features_df, distinct_activity_features_df)

    # print(activity_df)
    # print(full_activity_features_df['activity'].unique())
    # activity_df.to_csv(r'/Users/jankiefer/Documents/Studium/Master/Semester/5. Semester/RPA detector/output/{}.csv'.format(c.FILE_NAME), index=False, header=True)
    # full_activity_features_df.to_csv(r'/Users/jankiefer/Documents/Studium/Master/Semester/5. Semester/RPA detector/full.csv', index=False, header=True)
    # AutomationIndication.apply_bert(activity_df)
    df = pd.read_csv('./full.csv', delimiter=';')
    predictClass(df)


