from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
import torch
from tqdm import tqdm
from time import sleep
import evaluate
from accelerate import Accelerator
from colorama import Fore

class_dict = {
    1:"good",
    0:"bad"
}
#Helps with distributed computing
accelerator = Accelerator()

#Load Huggingface sentiment prediction
raw_datasets = load_dataset("glue", "sst2")
learner = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(learner)


def tokenize_function(example):
    #Tokenize an individual sentence from glue:sst2 according to distilbert-base-uncased tokenization
    return tokenizer(example["sentence"], truncation=True)

#Mapping function to cover all sentences in the dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

train_dataloader = DataLoader(
    tokenized_datasets["train"].shuffle(seed=42).select(range(1000)), shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"].shuffle(seed=42).select(range(500)), batch_size=8, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(learner, num_labels=2)
#Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
#Number of times to fine tune model on training set
num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

#Allow the program to take advantage of distributed computing
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

#Progress bar for training process
training_progress_bar = tqdm(range(num_training_steps), colour = "green", desc = "Training Progress: ")

#for i in tqdm(range(num_training_steps), colour="red"):
   # sleep(0.001)

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        #Create predictions for the particular embeddings for each iterative batch
        outputs = model(**batch)
        loss = outputs.loss
        #Helps with distributed computing
        accelerator.backward(loss)
        #Train at iterative step
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        training_progress_bar.update(1)
        training_progress_bar.refresh()
training_progress_bar.close()

metric = evaluate.load("glue", "sst2")
f1_score = evaluate.load("f1")
result = ""
eval_progress_bar = tqdm(range(len(eval_dataloader)), colour = "green", desc = "Evaluation Progress: ")
model.eval()
#Evaluation may be buggy at low amounts of training data
for batch in eval_dataloader:
    #Create predictions for test set
    with torch.no_grad():
        outputs = model(**batch)
    #Save Huggingface logit outputs
    logits = outputs.logits
    #Convert liklihoods to predictions
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    result = f1_score.compute(predictions=predictions, references=batch["labels"])
    eval_progress_bar.update(1)
    eval_progress_bar.refresh()
eval_progress_bar.close()
print(metric.compute())
print(result)


user_input = ""
print("Welcome to sentiment predictor. Enter your sentence, and wait for the predicted classification of negative [0], or positive [1].\n Use ^D to quit.\n")
#Take user input, i.e. sentences for sentiment prediction
while(input!="Quit"):
    user_input = input("Enter sentence >")
    encoding = tokenizer(user_input, return_tensors="pt")
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1)
    #print(str(type(predictions)))
    print(predictions.item())
    print(class_dict[int(predictions.item())])

text = ["this model is bad"]
encoding = tokenizer(text, return_tensors="pt")