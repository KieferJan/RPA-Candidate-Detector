import feature_extraction as fe
import bert_automation_indication.activity_bert_parser as AutomationIndication
import classifier.train as train
import classifier.predict as predict

# Settings can be defined in here
import constants as c


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Workflow Step - Feature Extraction
    default_df = fe.import_data()
    distinct_activity_features_df = fe.extract_activity_features(default_df)
    full_activity_features_df = fe.extract_activity_features_full_log(default_df)
    # join the two data frames (full trace into unique activities)
    activity_df = fe.join_full_in_distinct(full_activity_features_df, distinct_activity_features_df)

    # Workflow Step - Automation Indication Bert
    df = AutomationIndication.apply_bert(activity_df)

    # Workflow Step - Predict Class
    if c.DO_PREDICT_CLASSIFIER:
        print('Start predict')
        df = predict.start(df)
        print('Finish predict')
    else:
        print('Start train')
        train.start()
        df = predict.start(df)

    df.to_csv('./predictedDataset.csv', index=False, header=True)

