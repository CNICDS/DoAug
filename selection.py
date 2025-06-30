from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import pickle
from argparse import ArgumentParser
from utils import *

args = ArgumentParser()

args.add_argument("--dataset_name", type=str, default="anli")
args.add_argument("--seed", type=int, default=42)
args.add_argument("--reduce", action='store_true', default=False)
args.add_argument("--reduction_size", type=int, default=1200)
args = args.parse_args()

SEED = args.seed

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

def get_test_file(DATASET_NAME):
    if DATASET_NAME == "mnli":
        return 'validation.jsonl'
    else:
        return 'test.jsonl'

'''
quote (from IndexDataset to stratified_sampling)
data selection code from https://github.com/haizhongzheng/Coverage-centric-coreset-selection and https://github.com/adymaharana/d2pruning 
'''
class IndexDataset(torch.utils.data.Dataset):
    """
    The dataset also return index.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return idx, self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
class IndexedDataCollatorWithPadding(DataCollatorWithPadding):
    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"
    def __int__(self, tokenizer, padding, max_length, pad_to_multiple_of, return_tensors):
        super().__init__(tokenizer, padding, max_length, pad_to_multiple_of, return_tensors)
    def __call__(self, features):
        idxs = torch.tensor([idx for idx, _ in features])
        text_features = [f for _, f in features]
        batch = self.tokenizer.pad(
            text_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return (idxs, batch)
def post_training_metrics(model, dataloader, data_importance):
    model.eval()
    data_importance['entropy'] = torch.zeros(len(dataloader.dataset))
    data_importance['loss'] = torch.zeros(len(dataloader.dataset))
    data_importance['confidence'] = torch.zeros(len(dataloader.dataset))

    for iter, (idx, batch) in enumerate(tqdm(dataloader)):
        targets = batch['labels'].to('cuda')

        logits = model(**batch.to('cuda')).logits
        prob = nn.Softmax(dim=1)(logits)

        entropy = -1 * prob * torch.log(prob + 1e-10)
        entropy = torch.sum(entropy, dim=1).detach().cpu()

        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets).detach().cpu()
        confidence = prob[torch.arange(0, logits.shape[0]).to('cuda'), targets].detach().cpu()

        data_importance['entropy'][idx] = entropy
        data_importance['loss'][idx] = loss
        data_importance['confidence'][idx] = confidence
def training_dynamics_metrics(td_log, dataset, data_importance):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        data_idx, data = dataset[i]
        target = data['label']
        targets.append(target)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)
    data_importance['variance'] = []
    for i in range(data_size):
        data_importance['variance'].append([])

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        for i, idx in enumerate(index):
            data_importance['variance'][idx].append(target_prob[i].item())
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        record_training_dynamics(item)

    sizes = [len(data_importance['variance'][i]) for i in range(data_size)]
    for i, s in enumerate(sizes):
        if s != sizes[0]:
            for j in range(sizes[0] - s):
                data_importance['variance'][i].append(1.0)
    data_importance['variance'] = torch.tensor(np.std(np.array(data_importance['variance']), axis=-1))
def EL2N(td_log, dataset, data_importance, max_epoch=2, num_classes=None):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        data_idx, data = dataset[i]
        target = data['label']
        targets.append(target)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        el2n_score = torch.sqrt(l2_loss(label_onehot,output).sum(dim=1))

        data_importance['el2n'][index] += el2n_score

    for i, item in enumerate(td_log):
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item)
def score_monotonic_selection(data_score, key, ratio, descending, class_balanced):
    score = data_score[key]
    score_sorted_index = score.argsort(descending=descending)
    total_num = ratio * data_score[key].shape[0]
    print("Selecting from %s samples" % total_num)
    if class_balanced:
        print('Class balance mode.')
        all_index = torch.arange(data_score['targets'].shape[0])
        selected_index = []
        targets_list = data_score['targets'][score_sorted_index]
        targets_unique = torch.unique(targets_list)
        for target in targets_unique:
            target_index_mask = (targets_list == target)
            target_index = all_index[target_index_mask]
            targets_num = target_index_mask.sum()
            target_coreset_num = targets_num * ratio
            selected_index = selected_index + list(target_index[:int(target_coreset_num)])
            print("Selected %s samples for %s label" % (len(selected_index), target))
        selected_index = torch.tensor(selected_index)
        print(f'High priority {key}: {score[score_sorted_index[selected_index][:15]]}')
        print(f'Low priority {key}: {score[score_sorted_index[selected_index][-15:]]}')
        return score_sorted_index[selected_index]
    else:
        print(f'High priority {key}: {score[score_sorted_index[:15]]}')
        print(f'Low priority {key}: {score[score_sorted_index[-15:]]}')
        return score_sorted_index[:int(total_num)]
def mislabel_mask(data_score, mis_key, mis_num, mis_descending, coreset_key):
    mis_score = data_score[mis_key]
    mis_score_sorted_index = mis_score.argsort(descending=mis_descending)
    hard_index = mis_score_sorted_index[:mis_num]
    print(f'Bad data -> High priority {mis_key}: {data_score[mis_key][hard_index][:15]}')
    print(f'Prune {hard_index.shape[0]} samples.')

    easy_index = mis_score_sorted_index[mis_num:]
    data_score[coreset_key] = data_score[coreset_key][easy_index]

    return data_score, easy_index
def stratified_sampling(data_score, coreset_key, coreset_num):
        stratas = 50
        print('Using stratified sampling...')
        score = data_score[coreset_key]
        total_num = coreset_num

        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        step = (max_score - min_score) / stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)

        strata_num = torch.tensor(strata_num)

        def bin_allocate(num, bins):
            sorted_index = torch.argsort(bins)
            sort_bins = bins[sorted_index]

            num_bin = bins.shape[0]

            rest_exp_num = num
            budgets = []
            for i in range(num_bin):
                rest_bins = num_bin - i
                avg = rest_exp_num // rest_bins
                cur_num = min(sort_bins[i].item(), avg)
                budgets.append(cur_num)
                rest_exp_num -= cur_num


            rst = torch.zeros((num_bin,)).type(torch.int)
            rst[sorted_index] = torch.tensor(budgets).type(torch.int)

            return rst

        budgets = bin_allocate(total_num, strata_num)

        ##### sampling in each strata #####
        selected_index = []
        sample_index = torch.arange(data_score[coreset_key].shape[0])

        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]
            rand_index = torch.randperm(pool.shape[0])
            selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]

        return selected_index, None
'''end quote'''

DATASET_NAME = args.dataset_name

dataset = load_dataset(f"dataset-base/my-data/{DATASET_NAME}")
dataset = dataset["train"].shuffle(seed=SEED)
reduced_dataset_1200 = dataset.select(range(1200))
reduced_dataset_800 = dataset.select(range(800))
save_dataset_as_jsonl(reduced_dataset_1200, f"dataset-base/my-data/{DATASET_NAME}", "reduced_1200") # used for augmentation
save_dataset_as_jsonl(reduced_dataset_800, f"dataset-base/my-data/{DATASET_NAME}", "reduced_800") # used as "original" dataset / bsl aug / wo coreset

num_labels = get_num_labels(DATASET_NAME)
test_file = get_test_file(DATASET_NAME)
if args.reduce:
    dataset = load_dataset(f"dataset-base/my-data/{DATASET_NAME}",  data_files={
        'train': 'reduced_1200.jsonl',
    })
    reduce_prefix = ''
else:
    dataset = load_dataset(f"dataset-base/my-data/{DATASET_NAME}",  data_files={
        'train': 'train.jsonl',
    })
    reduce_prefix = 'full_'
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
indexed_data_collator = IndexedDataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
train_dataset = tokenized_dataset["train"]
train_dataset = IndexDataset(train_dataset)
train_loader = DataLoader(train_dataset, collate_fn=indexed_data_collator, batch_size=32, shuffle=False)
if not os.path.exists(f"selection/training_dynamics"):
    os.makedirs(f"selection/training_dynamics")
if not os.path.exists(f"selection/pretrained_models"):
    os.makedirs(f"selection/pretrained_models")
if not os.path.exists(f"selection/scores"):
    os.makedirs(f"selection/scores")
'''training dynamics'''
if not os.path.exists(f"selection/training_dynamics/{reduce_prefix}{DATASET_NAME}.pkl"):
    '''train to collect training dynamics'''
    print(f"Training on {reduce_prefix}{DATASET_NAME} for selection")
    training_dynamics = []
    model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH, num_labels=num_labels).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(3):
        model.train()
        for iter, (idx, batch) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            outputs = model(**batch.to("cuda"))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            td = {
                'epoch': epoch,
                'iteration': iter,
                'idx': idx,
                'output': F.log_softmax(outputs.logits, dim=-1).detach().cpu()
            }
            training_dynamics.append(td)
    with open(f"selection/training_dynamics/{reduce_prefix}{DATASET_NAME}.pkl", "wb") as f:
        pickle.dump(training_dynamics, f)
    # save model checkpoint
    model.save_pretrained(f"selection/pretrained_models/{reduce_prefix}{DATASET_NAME}")
else:
    '''load training dynamics'''
    print(f"Loading training dynamics for {reduce_prefix}{DATASET_NAME}")
    with open(f"selection/training_dynamics/{reduce_prefix}{DATASET_NAME}.pkl", "rb") as f:
        training_dynamics = pickle.load(f)
    model = AutoModelForSequenceClassification.from_pretrained(f"selection/pretrained_models/{reduce_prefix}{DATASET_NAME}").to("cuda")
'''scores'''
if not os.path.exists(f"selection/scores/{reduce_prefix}{DATASET_NAME}.pkl"):
    '''compute scores'''
    print(f"Computing importance scores for {reduce_prefix}{DATASET_NAME}")
    model.eval()
    data_score = {}
    post_training_metrics(model, train_loader, data_score)
    training_dynamics_metrics(training_dynamics, train_dataset, data_score)
    EL2N(training_dynamics, train_dataset, data_score, max_epoch=2, num_classes=num_labels)
    with open(f"selection/scores/{reduce_prefix}{DATASET_NAME}.pkl", "wb") as f:
        pickle.dump(data_score, f)
else:
    '''load scores'''
    print(f"Loading scores for {reduce_prefix}{DATASET_NAME}")
    with open(f"selection/scores/{reduce_prefix}{DATASET_NAME}.pkl", "rb") as f:
        data_score = pickle.load(f)
'''select half and augment'''
CORESET_RATIO = 0.5
MIS_RATIO = 0.1
MIS_KEY = 'accumulated_margin'
CORESET_KEYS = ['entropy', 'el2n', 'variance']
'''importance score selection'''
for coreset_key in CORESET_KEYS:
    coreset_index = score_monotonic_selection(
        data_score=data_score, key=coreset_key,
        ratio=CORESET_RATIO,
        descending=True,
        class_balanced=True)
    print(f"Selected {len(coreset_index)} samples for {coreset_key}")
    coreset = dataset['train'].select(coreset_index)
    print(coreset[0])
    save_dataset_as_jsonl(coreset, f"dataset-base/my-data/{DATASET_NAME}", f"{reduce_prefix}coreset_{coreset_key}_top_1_2")
'''ccs selection'''
CORESET_KEY = 'accumulated_margin'
total_num = len(dataset['train'])
coreset_num = int(CORESET_RATIO * total_num)
mis_num = int(MIS_RATIO * total_num)
data_score, score_index = mislabel_mask(data_score, mis_key=MIS_KEY, mis_num=mis_num, 
    mis_descending=MIS_KEY in ['entropy', 'forgetting', 'el2n', 'ssl'],
    coreset_key=CORESET_KEY)
coreset_index, _ = stratified_sampling(data_score=data_score, coreset_key=CORESET_KEY, coreset_num=coreset_num)
coreset_index = score_index[coreset_index]
coreset = dataset['train'].select(coreset_index)
print(coreset[0])
save_dataset_as_jsonl(coreset, f"dataset-base/my-data/{DATASET_NAME}", f"{reduce_prefix}coreset_ccs_aum_top_1_2")
'''select top and mid 1/3, drop bottom 1/3'''
for coreset_key in CORESET_KEYS:
    coreset_index_top_1_3 = score_monotonic_selection(
        data_score=data_score, key=coreset_key,
        ratio=1/3,
        descending=True,
        class_balanced=True)
    coreset_index_top_2_3 = score_monotonic_selection(
        data_score=data_score, key=coreset_key,
        ratio=2/3,
        descending=True,
        class_balanced=True)
    coreset_index_mid_1_3 = []
    for idx in range(len(coreset_index_top_2_3)):
        if coreset_index_top_2_3[idx] not in coreset_index_top_1_3:
            coreset_index_mid_1_3.append(coreset_index_top_2_3[idx])
    print(f"Selected {len(coreset_index_top_1_3)} samples for top 1/3 {coreset_key} and {len(coreset_index_mid_1_3)} samples for mid 1/3 {coreset_key}")
    coreset_top_1_3 = dataset['train'].select(coreset_index_top_1_3)
    coreset_mid_1_3 = dataset['train'].select(coreset_index_mid_1_3)
    coreset_top_2_3 = dataset['train'].select(coreset_index_top_2_3)
    print(coreset_top_1_3[0])
    print(coreset_mid_1_3[0])
    print(coreset_top_2_3[0])
    save_dataset_as_jsonl(coreset_top_1_3, f"dataset-base/my-data/{DATASET_NAME}", f"{reduce_prefix}coreset_{coreset_key}_top_1_3")
    save_dataset_as_jsonl(coreset_mid_1_3, f"dataset-base/my-data/{DATASET_NAME}", f"{reduce_prefix}coreset_{coreset_key}_mid_1_3")
    save_dataset_as_jsonl(coreset_top_2_3, f"dataset-base/my-data/{DATASET_NAME}", f"{reduce_prefix}coreset_{coreset_key}_top_2_3")
CORESET_KEY = 'accumulated_margin'
total_num = len(dataset['train'])
coreset_num_1_3 = int(1/3 * total_num)
coreset_num_2_3 = int(2/3 * total_num)
mis_num = int(MIS_RATIO * total_num)
data_score, score_index = mislabel_mask(data_score, mis_key=MIS_KEY, mis_num=mis_num, 
    mis_descending=MIS_KEY in ['entropy', 'forgetting', 'el2n', 'ssl'],
    coreset_key=CORESET_KEY)
coreset_index_top_1_3, _ = stratified_sampling(data_score=data_score, coreset_key=CORESET_KEY, coreset_num=coreset_num_1_3)
coreset_index_top_2_3, _ = stratified_sampling(data_score=data_score, coreset_key=CORESET_KEY, coreset_num=coreset_num_2_3)
coreset_index_top_1_3 = score_index[coreset_index_top_1_3]
coreset_index_top_2_3 = score_index[coreset_index_top_2_3]
coreset_index_mid_1_3 = []
for idx in range(len(coreset_index_top_2_3)):
    if coreset_index_top_2_3[idx] not in coreset_index_top_1_3:
        coreset_index_mid_1_3.append(coreset_index_top_2_3[idx])
print(len(coreset_index_top_1_3), len(coreset_index_top_2_3), len(coreset_index_mid_1_3))
print(f"Selected {len(coreset_index_top_1_3)} samples for top 1/3 ccs_aum and {len(coreset_index_mid_1_3)} samples for mid 1/3 ccs_aum")
coreset_top_1_3 = dataset['train'].select(coreset_index_top_1_3)
coreset_mid_1_3 = dataset['train'].select(coreset_index_mid_1_3)
coreset_top_2_3 = dataset['train'].select(coreset_index_top_2_3)
print(coreset_top_1_3[0])
print(coreset_mid_1_3[0])
print(coreset_top_2_3[0])
save_dataset_as_jsonl(coreset_top_1_3, f"dataset-base/my-data/{DATASET_NAME}", f"{reduce_prefix}coreset_ccs_aum_top_1_3")
save_dataset_as_jsonl(coreset_mid_1_3, f"dataset-base/my-data/{DATASET_NAME}", f"{reduce_prefix}coreset_ccs_aum_mid_1_3")
save_dataset_as_jsonl(coreset_top_2_3, f"dataset-base/my-data/{DATASET_NAME}", f"{reduce_prefix}coreset_ccs_aum_top_2_3")
