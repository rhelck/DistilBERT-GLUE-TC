from accelerate import Accelerator
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn import metrics
import numpy as np
import evaluate

accelerator = Accelerator()

dataset = load_dataset("glue","sst2")
print("\nA\n")
print(dataset["train"]['sentence'][1])
print("\nB\n")
num_labels = len(dataset['train'].features['label'].names)
print("\nC\n")
print(len(dataset['train']))
print(len(dataset['test']))
print("\nD\n")
#print(dataset["train"]["num_rows"])
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

AL_train_guess = round(len(dataset['train']) * .2)
#AL_test_guess

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(AL_train_guess))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

#Accelerator

model = accelerator.prepare(model)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()