import os

import numpy as np
import numpy.random as npr
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
)

pd.set_option("mode.copy_on_write", True)

dir = ""
grants = pd.read_parquet(dir + "subj_text_and_labels.parquet")

#########
# Split into train test and for predict
#########
grants = grants.drop_duplicates("description")
unlabeled = grants[grants["label_1"].isnull()]

# Load Model and Tokenizer
model = BertForSequenceClassification.from_pretrained(dir + "bert_grant_classifier")
tokenizer = BertTokenizer.from_pretrained(dir + "bert_grant_classifier")
label_encoder = torch.load(dir + "label_encoder.pth")

# Sample Input
sample_text = "AI Engineer from Researchify."
inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128,
).to(device)

# Prediction
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()
print(f"Predicted Section: {label_encoder.inverse_transform([predicted_class])[0]}")
