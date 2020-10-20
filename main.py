import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import feature_extraction as fe

def import_xes():
    pd.set_option('display.max_columns', None)
    pd.options.display.width = None
    log = xes_importer.apply('event logs/RequestForPayment.xes')
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    return df[['org:resource', 'concept:name', 'time:timestamp', 'org:role', 'case:Rfp_id']].copy()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    default_df = import_xes()
    distinct_activity_features_df = fe.extract_activity_features(default_df)
    full_activity_features_df = fe.extract_activity_features_full_log(default_df)

    # join the two different types of data frames (full trace into distinct activities AND
    # distinct activities into full trace)
    distinct_activity_features_df = fe.join_full_in_distinct(full_activity_features_df, distinct_activity_features_df)
    full_activity_features_df = fe.join_distinct_in_full(distinct_activity_features_df, full_activity_features_df)
    # print(distinct_activity_features_df)



