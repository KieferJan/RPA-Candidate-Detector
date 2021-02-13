import feature_extraction as fe
import bert_automation_indication.activity_bert_parser as AutomationIndication
import classifier.predict as predict
import time
import datetime


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # Workflow Step - Feature Extraction
    start = time.time()
    default_df = fe.import_data()
    distinct_activity_features_df = fe.extract_activity_features(default_df)
    full_activity_features_df = fe.extract_activity_features_full_log(default_df)

    #join the two data frames (aggregated trace numbers into unique activities)
    activity_df = fe.join_full_in_distinct(full_activity_features_df, distinct_activity_features_df)
    print('----Features Extracted Directly From Event Log----')

    # # Workflow Step - Automation Indication BERT
    df = AutomationIndication.apply_bert(activity_df)
    df.to_csv('./Output/extractedFeatures.csv', index=False, header=True)
    print('----BERT Confidence Features Extracted----')

    # Workflow Step - Predict Class
    print('Start predict')
    df = predict.start(df)
    print('Finish predict')
    df.to_csv('./Output/predictedDataset.csv', index=False, header=True)
    end = time.time()
    print(f'Elapsed time:{datetime.timedelta(seconds=end-start)}')



