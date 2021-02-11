# RPA-detector
- This project was developed on Python 3.7
- Install the requirements with "pip install requirements.txt"

## Download the required models
Due to limited storage on GitHub, please refer to the following Google Drive Storage download the respective models.

### For the bert_automation_indication directory:
This directory handles the extraction of the language based features. You can get access to the pyTorch Models via the link below.
https://drive.google.com/drive/folders/1MIR1Iap5p2Ap7M5_GZRp7E-bKjWd9dzW?usp=sharing

In total, three models need to be stored in the project directory "bert_automation_indication > Model > action | activity | business object"

### For the bert_parser directory:


Please store all the files in the projectdirectory "bert_parser" > "model"

## Project structure
Below is a brief introduction into the project structure.

### Directories
This tool consists of three larger components of which each is stored inside a directory:

- bert_automation_indication: This is the component that is responsible for the extraction of thje language-based feature
  - Model: A sudirectory that contains the three BERT models (action, business object, and activity)
- bert_parser: This is the component that is responsible for the semantic tags for the event label
- classifier: This component contains the classifier that predicts the RPA candidates

In addition to that, the following two directories contain the data that has been used along the project:

- event_logs_as_csv: contains the .xes event logs in csv format
- input_event_logs: contains the .xes event logs. If you want to predict a new event log, you have to store it in this directory.
- Outputs: Here, the output file "predictedDataset.csv" is stored. This data set contains the probability of the activity belonging to each of the four classes and, all the extracted features to provide further insights into the execution of an activity.

### Important files
Below is a brief description of the important files.

- Constants.py: This is the configuration file for the tools. It is seperated into three configuration areas:
  - Feature extraction: Here you can configure the feature extraction. For instance, the name of the event log that should be predicted, the names of the event log attributes for the respective event log, and the data type of the event log.
  - Automation_Indication: Here, the configuration of the language-based feature extraction is stored (default is to extract for type of action, type of business object, and activity)
  - Classifier: Here, the classifier is configured. Currently, it is configured for feature set 9.
- feature_extraction.py: This file contains the feature extraction of all features that can be extracted directly from the input event log
- main.py: Main file which defines the execution of the workflow
  - Extracts the preliminary data set features directly from the event log
  - Extracts the language-based features into the preliminary data set
  - Predicts tha RPA candidates based on the complete data set (Output into the directory "Output" > "predictedDataset.csv"


## Complementary paper and background
