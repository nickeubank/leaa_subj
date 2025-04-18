import os

import numpy as np
import numpy.random as npr
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (  # AdamW,
    BertForSequenceClassification,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

pd.set_option("mode.copy_on_write", True)

############
# Load data.
############

##########
# Colab
##########
# from google.colab import drive
# drive.mount('/content/gdrive/')
# dir = "/content/gdrive/MyDrive/leaa/"

#########
# Home
#########

dir = ""

########
# DCC
########

dir = "/hpc/group/ssri/nce8/leaa_subj"

########
# Cloud load
########
# dir = "https://github.com/nickeubank/leaa_subj/raw/refs/heads/main/"

#########
# Split into train test and for predict
#########

grants = pd.read_parquet(dir + "subj_text_and_labels.parquet")
grants = grants.drop_duplicates("description")

labeled = grants[grants["label_1"].notnull()]


# Encode labels. For 1 digit codes, not important, but
# the two digits aren't sequential so let's just use.
label_encoder = LabelEncoder()
labeled["label_1_encoded"] = label_encoder.fit_transform(labeled["label_1"])


labeled = labeled.sort_values("description")

train_label, test_label, train_text, test_text = train_test_split(
    labeled["label_1_encoded"].values,
    labeled["description"].values,
    test_size=0.2,
    random_state=42,
    stratify=labeled["label_1"],
)

########
# Preprocess
########


class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Define dataset
max_len = 128
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = ClassificationDataset(train_text, train_label, tokenizer, max_len)
test_dataset = ClassificationDataset(test_text, test_text, tokenizer, max_len)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model and Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=grants["label_1"].nunique()
)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
epochs = 1
for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch} Loss: {total_loss / len(train_loader)}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.4f}")

# Save
model.save_pretrained(dir + "bert_grant_classifier")
tokenizer.save_pretrained(dir + "bert_resume_classifier")
torch.save(label_encoder, dir + "label_encoder.pth")
