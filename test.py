from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased",padding=True)
#tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input)

decoded_input = tokenizer.decode(encoded_input["input_ids"])
print(decoded_input)

dataset = load_dataset("yelp_review_full")
dataset["train"][100]