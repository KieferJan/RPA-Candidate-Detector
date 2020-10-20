import pandas as pd
import bert_parser.main as bp
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
    event_df = import_xes()
    task_df = fe.extract_activity_labels(event_df)
    print(task_df)



