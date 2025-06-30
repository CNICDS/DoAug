from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import os
import pandas as pd
from argparse import ArgumentParser
from utils import *

args = ArgumentParser()

args.add_argument("--dataset_name", type=str, default="anli")
args.add_argument("--do_ori", action='store_true', default=False)
args.add_argument("--do_cst", action='store_true', default=False)
args.add_argument("--do_aug", action='store_true', default=False)
args.add_argument("--do_abl", action='store_true', default=False)
args.add_argument("--do_bsl", action='store_true', default=False)
args.add_argument("--do_var", action='store_true', default=False)
args = args.parse_args()
print(args)

if args.dataset_name in ["multi"]:
    BERT_PATH = "llm-base/google-bert/bert-base-multilingual-cased"
else:
    BERT_PATH = "llm-base/google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
metric = evaluate.load("evaluate/metrics/accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

SPLITS = get_splits()
USED_SPLITS = get_used_splits(
    do_ori=args.do_ori, do_cst=args.do_cst, do_aug=args.do_aug, do_abl=args.do_abl, do_bsl=args.do_bsl, do_var=args.do_var)

DATASET_NAME = args.dataset_name
print(f"Training on {DATASET_NAME}")
RESULTS_DIR = f"results/performance"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if os.path.exists(f"{RESULTS_DIR}/{DATASET_NAME}.csv"):
    dataframes = pd.read_csv(f"{RESULTS_DIR}/{DATASET_NAME}.csv", index_col=0)
else:
    dataframes = pd.DataFrame(columns=SPLITS)

num_labels = get_num_labels(DATASET_NAME)
test_file = get_test_file(DATASET_NAME)
CORESET_KEY = get_coreset_key(DATASET_NAME)
dataset = load_dataset(f"dataset-base/my-data/{DATASET_NAME}",  data_files=get_data_files(
    CORESET_KEY=CORESET_KEY, do_cst=args.do_cst, do_aug=args.do_aug, do_abl=args.do_abl, do_bsl=args.do_bsl, do_var=args.do_var, test_file=test_file))
tokenized_datasets = dataset.map(tokenize_function, batched=True)
eval_dataset = tokenized_datasets["test"]
if len(eval_dataset) > 4000:
    eval_dataset = eval_dataset.select(range(4000))
for split in USED_SPLITS:
    assert split in dataset.column_names

for split in USED_SPLITS:
    print(f"Training on {split}")
    train_dataset = tokenized_datasets[split]
    for seed in range(37,47):
        print(f"Train with seed {seed}")
        training_args = TrainingArguments(
            output_dir=f"trainer_saved/{DATASET_NAME}/{split}/seed_{seed}", 
            evaluation_strategy="no", # "epoch"
            seed=seed,
        )
        model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH, num_labels=num_labels)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        train_result = trainer.train()
        eval_result = trainer.evaluate()
        print(f"Dataset: {DATASET_NAME}, Split: {split}, Seed: {seed}, Acc: {eval_result['eval_accuracy']}")
        final_acc = eval_result['eval_accuracy']
        dataframes.loc[str(seed), split] = round(final_acc, 6)
        dataframes.to_csv(f"{RESULTS_DIR}/{DATASET_NAME}.csv")
        
    split_avg_acc = dataframes[split].mean(skipna=True).round(4)
    split_std_acc = dataframes[split].std(skipna=True).round(4)
    dataframes.loc["avg", split] = split_avg_acc
    dataframes.loc["std", split] = split_std_acc

dataframes = dataframes[SPLITS]
dataframes.to_csv(f"{RESULTS_DIR}/{DATASET_NAME}.csv")




