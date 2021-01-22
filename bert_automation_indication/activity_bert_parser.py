import torch
import pandas as pd
import logging

from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
import constants as c

def apply_bert(df):
    logging.basicConfig(level=logging.ERROR)
    original_df = df.copy()
    for col in c.TEXT_COLUMNS:
        prep_df = preprocess(df, col)
        prep_df = prep_df.drop_duplicates(subset=[col])
        result_df = predict(col, prep_df)
        original_df = original_df.join(result_df.set_index(col), on=col)
    return original_df


def preprocess(orig_df, col):
    df = orig_df.copy()
    df.drop(df.columns.difference([col, 'task_type']), 1, inplace=True)

    # remove missing values
    df = df[df[col] != 'missing']

    # Map class label
    if col != 'action':
         df['task_type'] = df['task_type'].map(
            {'Physical or Cognitive Task': 'Physical or Cognitive Task', 'Low Automatable User Task': 'Automated',
            'High Automatable User Task': 'Automated', 'Automated': 'Automated'})

    unique_labels = df['task_type'].unique()
    label_dict = {}
    for index, unique_label in enumerate(unique_labels):
         label_dict[unique_label] = index
    df['label'] = df['task_type'].replace(label_dict)

    return df

def predict(col, df):
    print('Start extract BERT features')

    if col == 'action':
        label_dict = c.LABEL_DICT_ACTION
    else:
        label_dict = c.LABEL_DICT

    label_dict_inverse = {v: k for k, v in label_dict.items()}
    headers = []
    for key in label_dict_inverse:
        headers.append(f'Confidence_{col}_{label_dict_inverse[key]}')

    device = torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained(f'./bert_automation_indication/Model/{col}/model/')
    tokenizer = BertTokenizer.from_pretrained(f'./bert_automation_indication/Model/{col}/model/')
    model.to(device)
    model.eval()
    print('Fine-tuned model loaded')
    encoded_data_pred = tokenizer.batch_encode_plus(
        df[col].values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_pred = encoded_data_pred['input_ids']
    attention_masks_pred = encoded_data_pred['attention_mask']
    labels_pred = torch.tensor(df.label.values)
    dataset_pred = TensorDataset(input_ids_pred, attention_masks_pred, labels_pred)

    batch_size = c.BATCH_SIZE

    dataloader_pred = DataLoader(dataset_pred,
                                 sampler=SequentialSampler(dataset_pred),
                                 batch_size=batch_size)

    preds, true_vals = [], []
    for batch in dataloader_pred:
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs[1]

        logits = logits.numpy()
        preds.append(F.softmax(torch.tensor(logits), dim=1))


    preds = torch.cat(preds, 0)
    result_df = pd.DataFrame(preds, columns=headers).astype("float")
    df.reset_index(inplace=True)
    result_df[col] = df[col]
    print(f'Finished Feature Extraction for {col}')
    return result_df
