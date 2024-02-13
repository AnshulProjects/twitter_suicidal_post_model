import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel,DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from transformers import Trainer,TrainingArguments
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np
import evaluate


'''
Preprocessing CSV dataset
'''
df = pd.read_csv("/Users/anshulprasad/Documents/twitter_suicidal_post_model/Suicide_Ideation_Dataset(Twitter-based).csv")

df["Suicide"] = [0 if x == "Not Suicide post" else 1 for x in df["Suicide"]]

df["Tweet"] = ["Is this post potentially suicidal: " + str(x) for x in df["Tweet"]]

df = df.rename(columns={"Tweet": "text", "Suicide": "label"})

data_set_train =  Dataset.from_pandas(df,split=['train', 'test'])

data_set = data_set_train.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


'''
tokenizing tweets for the LLM.
'''
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = data_set.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",num_train_epochs=1)

metric = evaluate.load("accuracy")


'''
Metrics for training
'''

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

trainer.save_model("./model")

trainer.train()

