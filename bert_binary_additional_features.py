import argparse
import pandas as pd
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix


#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#Transformers library for BERT
import transformers
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import classification_report, confusion_matrix

#Seed for reproducibility
import random

seed_value=42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

import time
MAX_LEN = 128



class Bert_Classifier(torch.nn.Module):
    def __init__(self, freeze_bert=False, input_shape=(768, 50, 9, 2)):
        super(Bert_Classifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.cnt = 0
        H, hidden_layers, additional, num_of_classes = input_shape

        # MLP
        self.linear1 = nn.Linear(H, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers + additional, num_of_classes)
        
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.relu = nn.ReLU()

        # Add possibility to freeze the BERT model
        # to avoid fine tuning BERT params (usually leads to worse results)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, other_features):
        # Feed input data to BERT
        bert_outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = bert_outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        
        first_linear_outputs = self.drop(self.relu(self.linear1(last_hidden_state_cls)))

        input_for_linear2 = torch.cat((first_linear_outputs, other_features), 1)

        logits = self.drop(self.relu(self.linear2(input_for_linear2)))

        return logits

def train(model, train_dataloader, val_dataloader=None, epochs=3, batch_size=32, verbose=False, evaluation=False):
    # Define Cross entropy Loss function for the multiclass classification task
    loss_fn = nn.CrossEntropyLoss()

    print("Start training...\n")

    acc_through_batches_val = []

    for epoch_i in range(epochs):

        print("Epoch : {}".format(epoch_i+1))
        print("-"*10)
        print("-"*38)
        print(f"{'BATCH NO.':^7} | {'TRAIN LOSS':^12} | {'ELAPSED (s)':^9}")
        print("-"*38)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        #training mode
        model.train()

        for batch_idx, batch in enumerate(train_dataloader):
            
            b_input_ids, b_attn_mask, b_additional_inputs, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            #forward pass 
            logits = model(b_input_ids, b_attn_mask, b_additional_inputs)

            # Compute loss and accumulate the loss values

            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            #backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update model parameters:
            # fine tune BERT params and train additional dense layers
            optimizer.step()

            # update learning rate
            #scheduler.step()

            # Print the loss values and time elapsed for every 100 batches
            if (batch_idx % 100 == 0 and batch_idx != 0) or (batch_idx == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                print(f"{batch_idx:^9} | {batch_loss / (100 if batch_idx % 100 == 0 else batch_idx % 100)  :^12.6f} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss = 0
                t0_batch = time.time()

        # Calculate the average loss in epoch
        avg_train_loss = total_loss / len(train_dataloader)

        ###EVALUATION###
        
        # Put the model into the evaluation mode
        model.eval()
        
        # Define empty lists to host accuracy and validation for each batch
        val_accuracy = []
        val_loss = []

        for batch in val_dataloader:
            batch_input_ids, batch_attention_mask, batch_additional_inputs, batch_labels = tuple(t.to(device) for t in batch)
            
            # We do not want to update the params during the evaluation,
            # So we specify that we dont want to compute the gradients of the tensors
            # by calling the torch.no_grad() method
            with torch.no_grad():
                logits = model(batch_input_ids, batch_attention_mask, batch_additional_inputs)

            loss = loss_fn(logits, batch_labels)

            val_loss.append(loss.item())

            # Get the predictions starting from the logits (get index of highest logit)
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the validation accuracy 
            accuracy = (preds == batch_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)


        acc_through_batches_val.extend(val_accuracy)

        # Compute the average accuracy and loss over the validation set
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)
        


        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch
        print("-"*61)
        print(f"{'AVG TRAIN LOSS':^12} | {'VAL LOSS':^10} | {'VAL ACCURACY (%)':^9} | {'ELAPSED (s)':^9}")
        print("-"*61)
        print(f"{avg_train_loss:^14.6f} | {val_loss:^10.6f} | {val_accuracy:^17.2f} | {time_elapsed:^9.2f}")
        print("-"*61)
        print("\n")

        torch.save(bert_classifier, f'./bert_binary_model_w_add_f_epoch_{epoch_i}_acc_{val_accuracy}.pt')
    
    print("Training complete!")

    if verbose:
        print("\n\n")
        print("acc_through_batches_val =", acc_through_batches_val)


#tokenization
def bert_tokenizer(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []
    attention_masks = []
    for tweet in data:
        encoded_tweet = tokenizer.encode_plus(
            text=tweet,
            max_length=MAX_LEN,             # Choose max length to truncate/pad
            truncation=True,
            pad_to_max_length=True,         # Pad sentence to max length 
            return_attention_mask=True      # Return attention mask
        )
        input_ids.append(encoded_tweet.get('input_ids'))
        attention_masks.append(encoded_tweet.get('attention_mask'))
    
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks



def prepare_data(train_data, val_data, batch_size=32):
    df_t = pd.read_csv(train_data)
    df_v = pd.read_csv(val_data)

    X_train = df_t['text'].values
    y_train = df_t['class'].values
    X_additional_features_train = df_t[[ 'length', 'emoji_count',
       'tags_count', 'sentence_count', 'url_count',
       'number_count', 'punctuation_count', 'quote_count',
       'upper_word_count']].to_numpy()

    X_val = df_v['text'].values
    y_val = df_v['class'].values
    X_additional_features_val = df_v[[ 'length', 'emoji_count',
       'tags_count', 'sentence_count', 'url_count',
       'number_count', 'punctuation_count', 'quote_count',
       'upper_word_count']].to_numpy()

    train_inputs, train_masks = bert_tokenizer(X_train)
    val_inputs, val_masks = bert_tokenizer(X_val)

    train_additional_inputs = torch.from_numpy(X_additional_features_train)
    val_additional_inputs = torch.from_numpy(X_additional_features_val)

    train_labels = torch.from_numpy(y_train)
    val_labels = torch.from_numpy(y_val)

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_additional_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_additional_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 3
    batch_size = 32
    lr = 5e-5

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str,
                        help='path to .csv file with train data')
    parser.add_argument('--val_data', type=str,
                        help='path to .csv file with validation data')
    parser.add_argument('--epochs', type=str,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=str,
                        help='batch size number for training')
    parser.add_argument('--lr', type=str,
                        help='learning rate for training')
    parser.add_argument('--verbose', action="store_true",
                        help='if you want training details')
    parser.add_argument('--saved_model', type=str,
                        help='to continue training on last saved model; import saved model') 
    args = parser.parse_args()

    if args.train_data:
        train_data = args.train_data
    if args.val_data:
        val_data = args.val_data
    if args.epochs:
        epochs = int(args.epochs)
    if args.batch_size:
        batch_size = int(args.batch_size)
    if args.lr:
        lr = float(args.lr)
    verbose = True if args.verbose else False

    train_dataloader, val_dataloader = prepare_data(train_data, val_data, batch_size)

    if args.saved_model:
        saved_model = args.saved_model
        bert_classifier = torch.load(saved_model)
    else:
        bert_classifier = Bert_Classifier(freeze_bert=False)
    
    bert_classifier.to(device)

    # Set up optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=lr,    # learning rate, set to default value
                      #eps=1e-8    # decay, set to default value
                      )
    
    ### Set up learning rate scheduler ###

    # Calculate total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Defint the scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)

    train(bert_classifier, train_dataloader, val_dataloader, epochs=epochs, batch_size = batch_size, verbose = verbose)
    torch.save(bert_classifier, './bert_binary_w_add_model.pt')