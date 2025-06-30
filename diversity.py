from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
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
model = AutoModelForMaskedLM.from_pretrained(BERT_PATH).to("cuda")
model.eval()

def calculate_homogeneity(embeddings):
    mean_dot_prod_sim = []
    m = torch.tensor(embeddings.size(0))
    H = torch.tensor(embeddings.size(1))
    entropy = torch.tensor(0.0).cpu()
    upper_bound = torch.log(m-1).cpu()
    for embed in tqdm(embeddings):
        diff_ij = embed - embeddings
        squared_dist_ij = torch.square(diff_ij).sum(dim=1)
        weights_ij = torch.sqrt(squared_dist_ij) ** torch.log(H)
        sum_weights_ik = weights_ij.sum()
        prob_trans_ij = weights_ij / sum_weights_ik
        log_prob_trans_ij = torch.log(prob_trans_ij+1e-10)
        v = 1 / prob_trans_ij.size(0)
        entropy_ij = - torch.sum(v * prob_trans_ij * log_prob_trans_ij).cpu()
        entropy = entropy + entropy_ij
        mean_dot_prod_sim.append(torch.mean(weights_ij.detach(), dim=0))
    homogeneity = (entropy/upper_bound).cpu().item()
    return homogeneity
def preprocess(text):
    text = text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '')
    return text.split()

RESULTS_DIR = f"results/diversity"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

DATASET_NAME = args.dataset_name
CORESET_KEY = get_coreset_key(DATASET_NAME)
SPLITS = get_splits()
USED_SPLITS = get_used_splits(
    do_ori=args.do_ori, do_cst=args.do_cst, do_aug=args.do_aug, do_abl=args.do_abl, do_bsl=args.do_bsl, do_var=args.do_var, CORESET_KEY=CORESET_KEY)
test_file = get_test_file(DATASET_NAME)
dataset = load_dataset(f"dataset-base/my-data/{DATASET_NAME}",  data_files=get_data_files(
    CORESET_KEY=CORESET_KEY, do_cst=args.do_cst, do_aug=args.do_aug, do_abl=args.do_abl, do_bsl=args.do_bsl, do_var=args.do_var, test_file=test_file, use_clean=True))
tokenized_datasets = dataset.map(tokenize_function, batched=True)
def pad_to_max_length(examples):
    return tokenizer.pad(examples, padding='longest')
tokenized_datasets = tokenized_datasets.map(pad_to_max_length)
tensor_dataset = tokenized_datasets.remove_columns(["text"]).with_format("torch")

# %% result files
if os.path.exists(f"{RESULTS_DIR}/class_affinity.csv"):
    df_class_affinity = pd.read_csv(f"{RESULTS_DIR}/class_affinity.csv", index_col=0)
else:
    df_class_affinity = pd.DataFrame(columns=SPLITS)
if os.path.exists(f"{RESULTS_DIR}/class_distance.csv"):
    df_class_distance = pd.read_csv(f"{RESULTS_DIR}/class_distance.csv", index_col=0)
else:
    df_class_distance = pd.DataFrame(columns=SPLITS)
if os.path.exists(f"{RESULTS_DIR}/class_dissimilarity.csv"):
    df_class_dissimilarity = pd.read_csv(f"{RESULTS_DIR}/class_dissimilarity.csv", index_col=0)
else:
    df_class_dissimilarity = pd.DataFrame(columns=SPLITS)
if os.path.exists(f"{RESULTS_DIR}/class_embed_std.csv"):
    df_class_embed_std = pd.read_csv(f"{RESULTS_DIR}/class_embed_std.csv", index_col=0)
else:
    df_class_embed_std = pd.DataFrame(columns=SPLITS)
if os.path.exists(f"{RESULTS_DIR}/class_homogeneity.csv"):
    df_class_homogeneity = pd.read_csv(f"{RESULTS_DIR}/class_homogeneity.csv", index_col=0)
else:
    df_class_homogeneity = pd.DataFrame(columns=SPLITS)
if os.path.exists(f"{RESULTS_DIR}/token_count.csv"):
    df_token_count = pd.read_csv(f"{RESULTS_DIR}/token_count.csv", index_col=0)
else:
    df_token_count = pd.DataFrame(columns=SPLITS)
if os.path.exists(f"{RESULTS_DIR}/unique_3grams.csv"):
    df_unique_3grams = pd.read_csv(f"{RESULTS_DIR}/unique_3grams.csv", index_col=0)
else:
    df_unique_3grams = pd.DataFrame(columns=SPLITS)


# %% token level diversity
for split in USED_SPLITS:
    unique_3grams = set()
    for example in dataset[split]:
        text = example['text']
        tokens = preprocess(text)
        ngrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
        unique_3grams.update(ngrams)
    df_unique_3grams.loc[DATASET_NAME, split] = len(unique_3grams)
    df_unique_3grams.to_csv(f"{RESULTS_DIR}/unique_3grams.csv")
for split in USED_SPLITS:
    token_set = set()
    loader = DataLoader(tensor_dataset[split], batch_size=16)
    for batch in tqdm(loader):
        for sample in batch["input_ids"]:
            for token in sample:
                token_set.add(token.item())
    df_token_count.loc[DATASET_NAME, split] = len(token_set)
    df_token_count.to_csv(f"{RESULTS_DIR}/token_count.csv")
# %% embed samples, and arrange samples and centers by class 
split_class_samples = {} # key: split, value: dict of class samples
split_class_centers = {} # key: split, value: dict of class centers
for split in USED_SPLITS:
    class_samples = {} # key: class, value: a tensor of N*H where N is the number of samples in a class and H is the hidden size
    class_centers = {} # key: class, value: a tensor of center of the class
    print(f"embedding for {split}")
    split_embeddings = torch.tensor([], device="cuda")
    loader = DataLoader(tensor_dataset[split], batch_size=16)
    for batch in tqdm(loader):
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"].to('cuda'), attention_mask=batch["attention_mask"].to('cuda'), return_dict=True, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][:, 0, :]
        labels = batch["label"]
        for sample_idx, sample in enumerate(batch["input_ids"]):
            label = labels[sample_idx].item()
            if label not in class_samples:
                class_samples[label] = embeddings[sample_idx].unsqueeze(0)
            else:
                class_samples[label] = torch.cat((class_samples[label], embeddings[sample_idx].unsqueeze(0)))
    for label in class_samples:
        class_centers[label] = class_samples[label].mean(dim=0).unsqueeze(0)
    split_class_samples[split] = class_samples
    split_class_centers[split] = class_centers
# %% calculate diversity metrics
for split in USED_SPLITS:
    print(f"Computing diversity for {split}")
    class_avg_distances = [] # avg distance between samples in the same class
    class_avg_dissimilarities = [] # 1/similarity between samples in the same class
    class_embed_stds = [] # radius of the class (mean along the hidden size)
    class_homogeneities = [] # homogeneity of the class
    for label in split_class_samples[split].keys():
        samples = split_class_samples[split][label]
        avg_distance = torch.cdist(samples, samples).mean().cpu().item()
        avg_dissimilarity = 1 - torch.nn.functional.cosine_similarity(samples.unsqueeze(1), samples.unsqueeze(0), dim=-1).mean().cpu().item()
        embed_std = (torch.prod(torch.std(samples, dim=1)) ** (1 / samples.size(0))).cpu().item()
        if embed_std == 0: # if zero, use arithmetic mean (probably due to precision error)
            embed_std = torch.std(samples, dim=1).mean().cpu().item()
        homogeneity = calculate_homogeneity(samples)
        
        class_avg_distances.append(avg_distance)
        class_avg_dissimilarities.append(avg_dissimilarity)
        class_embed_stds.append(embed_std)
        class_homogeneities.append(homogeneity)

    df_class_distance.loc[DATASET_NAME, split] = round(np.mean(class_avg_distances), 6)
    df_class_dissimilarity.loc[DATASET_NAME, split] = round(np.mean(class_avg_dissimilarities), 6)
    df_class_embed_std.loc[DATASET_NAME, split] = round(np.mean(class_embed_stds), 6)
    df_class_homogeneity.loc[DATASET_NAME, split] = round(np.mean(class_homogeneities), 6)

    df_class_distance.to_csv(f"{RESULTS_DIR}/class_distance.csv")
    df_class_dissimilarity.to_csv(f"{RESULTS_DIR}/class_dissimilarity.csv")
    df_class_embed_std.to_csv(f"{RESULTS_DIR}/class_embed_std.csv")
    df_class_homogeneity.to_csv(f"{RESULTS_DIR}/class_homogeneity.csv")
# %% calculate affinity and fbd
for split in USED_SPLITS:
    print(f"Computing affinity for {split}")
    # set to 0, use +=, and /len() is equal to set to [], use append, and mean
    deviation = 0
    if split in ['dpo_d', 'dpo_s', 'sft_d', 'sft_s', 'wo_coreset_dpo_d', 'wo_selective_dpo_d', '10k', '20k', '50k', '100k', 't1.2', 'prompt']:
        ref_split = f'{CORESET_KEY}_top_2_3'
    else: 
        ref_split = 'reduced_800'
    ref_split = f'{CORESET_KEY}_top_2_3'
    for label in split_class_centers[split]:
        ## affinity
        deviation += torch.norm(split_class_centers[split][label] - split_class_centers[ref_split][label])
    deviation /= len(split_class_centers[split])
    affinity = 1 / deviation
    df_class_affinity.loc[DATASET_NAME, split] = round(affinity.item(), 6)
    df_class_affinity.to_csv(f"{RESULTS_DIR}/class_affinity.csv")

