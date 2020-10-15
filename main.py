import pandas as pd
import bert_parser.main as bp
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer

def import_xes():
    pd.set_option('display.max_columns', None)
    pd.options.display.width = None
    log = xes_importer.apply('event logs/RequestForPayment.xes')
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    return df[['org:resource', 'concept:name', 'time:timestamp', 'org:role', 'case:Rfp_id']].copy()

def extract_activity_labels(df):
    events = df['concept:name']
    tagged_events = bp.main(events)
    for key in tagged_events:
        print(key, '->', tagged_events[key])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = import_xes()
    extract_activity_labels(df)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
