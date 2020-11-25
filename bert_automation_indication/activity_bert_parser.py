import torch
import pandas as pd

from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import random
import numpy as np
import torch.nn.functional as F

text_columns = ['business object'] #, 'business object', 'action']


def apply_bert(df):
    do_train = False
    do_predict = True
    original_df = df.copy()
    for col in text_columns:
        prep_df, label_dict = preprocess(df, col)
        prep_df = prep_df.drop_duplicates(subset=[col])
        if do_train:
            dataloader_validation = train(prep_df, col, label_dict, True)
            test(label_dict, dataloader_validation, col)
        if do_predict:
            # prep_df = prep_df.drop_duplicates(subset=[col])
            result_df = predict(col, prep_df)
            original_df = original_df.join(result_df.set_index(col), on=col)
            original_df.to_csv(f'/Users/jankiefer/Documents/Studium/Master/Semester/5. Semester/RPA detector/'
                               f'bert_automation_indication/Output/full_{col}.csv', index=False)


def preprocess(df, col):
    df.drop(df.columns.difference([col, 'task_type']), 1, inplace=True)

    # remove missing values
    df = df[df[col] != 'missing']

    # Map class label
    df['task_type'] = df['task_type'].map(
        {'Physical or Cognitive Task task': 'Physical or Cognitive Task', 'Low Automatable User Task': 'Automated',
         'High Automatable User Task': 'Automated', 'Automated': 'Automated'})
    unique_labels = df['task_type'].unique()

    label_dict = {}
    for index, unique_label in enumerate(unique_labels):
        label_dict[unique_label] = index

    df['label'] = df['task_type'].replace(label_dict)
    return df, label_dict


def train(df, col, label_dict, do_train):
    X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                      df.label.values,
                                                      test_size=0.15,
                                                      random_state=42,
                                                      stratify=df.label.values)

    df['data_type'] = ['not_set'] * df.shape[0]

    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'

    df.groupby(['task_type', 'label', 'data_type']).count()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                               do_lower_case=True)

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == 'train'][col].values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == 'val'][col].values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type == 'train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type == 'val'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    batch_size = 16

    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)

    optimizer = AdamW(model.parameters(),
                      lr=1e-5,
                      eps=1e-8)

    epochs = 5

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    for epoch in range(1, epochs + 1):

        model.train()

        loss_train_total = 0

        # progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in dataloader_train:
            model.zero_grad()

            # batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
        torch.save(model.state_dict(), f'bert_automation_indication/Model/{col}/finetuned_BERT_epoch_{epoch}.model')

        # tqdm.write(f'\nEpoch {epoch}')
        print(f'\n Train Epoch: {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        # tqdm.write(f'Training loss: {loss_train_avg}')
        print(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals, log_probs = evaluate(dataloader_validation, model)
        val_f1 = f1_score_func(predictions, true_vals)
        # tqdm.write(f'Validation loss: {val_loss}')
        # tqdm.write(f'F1 Score (Weighted): {val_f1}')
        print(f'Validation loss: {val_loss}')
        print(f'F1 Score (Weighted): {val_f1}')
        print(f'Finished Epoch: {epoch}')

    return dataloader_validation


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')


def evaluate(dataloader_val, model):
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model.eval()

    loss_val_total = 0
    predictions, true_vals, log_probs = [], [], []

    for batch in dataloader_val:
        # batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        log_probs.append(F.softmax(torch.tensor(logits), dim=1))
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)
    log_probs = np.concatenate(log_probs, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals, log_probs


def test(label_dict, dataloader_validation, col):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    model.to('cpu')
    for i in range(1, 6):
        model.load_state_dict(torch.load(f'bert_automation_indication/Model/{col}/finetuned_BERT_epoch_{i}.model',
                                     map_location=torch.device('cpu')))

        _, predictions, true_vals, log_probs = evaluate(dataloader_validation, model)
        accuracy_per_class(predictions, true_vals, label_dict)


def predict(col, df):
    label_dict = {'Automated': 0, 'Physical or Cognitive Task': 1}
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    headers = []
    for key in label_dict_inverse:
        headers.append(f'Confidence_{col}_{label_dict_inverse[key]}')

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=2,
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    model.to('cpu')

    model.load_state_dict(
        torch.load(f'./bert_automation_indication/Model/{col}/finetuned_BERT.model',
                   map_location=torch.device('cpu')))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)

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

    batch_size = 16
    dataloader_pred = DataLoader(dataset_pred,
                                 sampler=SequentialSampler(dataset_pred),
                                 batch_size=batch_size)

    preds = []
    for batch in dataloader_pred:
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs[1]

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        preds.append(F.softmax(torch.tensor(logits), dim=1))

    preds = torch.cat(preds, 0)
    result_df = pd.DataFrame(preds, columns=headers).astype("float")
    df.reset_index(inplace=True)
    result_df[col] = df[col]

    return result_df
