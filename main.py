import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import feature_extraction as fe
import constants as c
xes_log = ""


def import_xes():

    pd.set_option('display.max_columns', None)
    pd.options.display.width = None
    global xes_log
    xes_log = xes_importer.apply('event logs/{}.xes'.format(c.FILE_NAME))
    df = log_converter.apply(xes_log, variant=log_converter.Variants.TO_DATA_FRAME)
    attribute_list_copy = c.ATTRIBUTE_LIST
    for attribute in attribute_list_copy:
        if attribute not in df.columns:
            attribute_list_copy.remove(attribute)
    df[c.ORG_RESOURCE_ATTRIBUTE_NAME].fillna('missing', inplace=True)
    return df[attribute_list_copy]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    default_df = import_xes()
    distinct_activity_features_df = fe.extract_activity_features(default_df, xes_log)
    full_activity_features_df = fe.extract_activity_features_full_log(default_df)

    # join the two different types of data frames (full trace into distinct activities AND
    # distinct activities into full trace)
    activity_df = fe.join_full_in_distinct(full_activity_features_df, distinct_activity_features_df)

    print(activity_df)
    print(full_activity_features_df)
    activity_df.to_csv(r'/Users/jankiefer/Documents/Studium/Master/Semester/5. Semester/RPA detector/output/{}.csv'.format(c.FILE_NAME), index=False, header=True)
    # full_activity_features_df.to_csv(r'/Users/jankiefer/Documents/Studium/Master/Semester/5. Semester/RPA detector/full.csv', index=False, header=True)



