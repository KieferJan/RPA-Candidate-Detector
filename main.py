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
    bo = []
    action = []
    actor = []
    act = []
    for activity in tagged_events:
        act.append(activity)
        i = 0
        bo_indexes = []
        a_indexes = []
        actor_indexes = []
        #retrieve index of BO,A, ACTOR tags
        for tag in tagged_events[activity]:
            if tag == "BO":
                bo_indexes.append(i)
            elif tag == "A":
                a_indexes.append(i)
            elif tag == "ACTOR":
                actor_indexes.append(i)
            i += 1
        print(f"{activity} BO: {bo_indexes} A: {a_indexes} ACTOR: {actor_indexes}")
        activity_list_words = activity.split( )
        # iterate through all words of an activity and identify the BO, A, Actor values and create a separate list for
        # each tag
        bo_values = []
        for b in bo_indexes:
            bo_values.append(activity_list_words[b])
        bo.append(" ".join(bo_values))

        a_values = []
        for a in a_indexes:
            a_values.append(activity_list_words[a])
        action.append(" ".join(a_values))

        actor_values = []
        for ac in actor_indexes:
            actor_values.append(activity_list_words[ac])
        actor.append(" ".join(actor_values))

    dict = {"activity": act, "business object": bo, "action": action, "resource": actor}
    result_df = pd.DataFrame(dict)
    return result_df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = import_xes()
    extract_activity_labels(df)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
