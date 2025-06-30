'''
prepare the paraphrase dataset for llama finetuning (sft+dpo)
the original dataset is from https://huggingface.co/datasets/humarin/chatgpt-paraphrases
'''

from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
import torch
import os
import random
import json
import ast
from tqdm import tqdm

if not os.path.exists(os.path.join('dataset-base', 'my-data', 'paraphrases')):
    os.makedirs(os.path.join('dataset-base', 'my-data', 'paraphrases'))

paraphrase_dataset = load_dataset("dataset-base/humarin/chatgpt-paraphrases", split="train").shuffle(seed=42)
question_dataset = paraphrase_dataset.filter(lambda x: x["category"] == "question")
sentence_dataset = paraphrase_dataset.filter(lambda x: x["category"] == "sentence")
QUESTION_PROPORTION = 0.05 # initial paraphrase dataset contains 59% question and 41% sentence, but not that many questions in the real-world datasets

SFT_SIZE = 100000//5 # 5 paraphrases per sentence, 20k original -> 100k paraphrases
DPO_SIZE = 100000 # 100k preference pairs 
SFT_SUFFIX = f"{SFT_SIZE*5//1000}k"
DPO_SUFFIX = f"{DPO_SIZE//1000}k"

def process_sft():
    paraphrase_sft = []
    question_dst = question_dataset.select(range(int(SFT_SIZE * QUESTION_PROPORTION)))
    sentence_dst = sentence_dataset.select(range(int(SFT_SIZE * (1 - QUESTION_PROPORTION))))
    for dst in [question_dst, sentence_dst]:
        for sample in tqdm(dst):
            instruction = "You will be given a sentence. Please paraphrase the sentence. "
            input = sample["text"]
            paraphrases = ast.literal_eval(sample["paraphrases"]) # "['paraphrased text 1', 'paraphrased text 2', ...]" -> ['paraphrased text 1', 'paraphrased text 2', ...]
            for paraphrase in paraphrases:
                paraphrase_sft.append({
                    "instruction": instruction,
                    "input": input,
                    "output": paraphrase
                })
    random.shuffle(paraphrase_sft, random.seed(42))
    with open(f"dataset-base/my-data/paraphrases/paraphrase_sft_{SFT_SUFFIX}_{DPO_SUFFIX}.json", "w") as f:
        json.dump(paraphrase_sft, f, indent=2)

process_sft()

def process_dpo():
    bert_path = "llm-base/google-bert/bert-base-uncased"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path).to("cuda")
    bert_model.to("cuda")
    bert_model.eval()

    paraphrase_dpo = []
    question_dst = question_dataset.select(range(int(SFT_SIZE), int(SFT_SIZE + DPO_SIZE * QUESTION_PROPORTION)))
    sentence_dst = sentence_dataset.select(range(int(SFT_SIZE), int(SFT_SIZE + DPO_SIZE * (1 - QUESTION_PROPORTION))))
    for dst in [question_dst, sentence_dst]:
        for sample in tqdm(dst):
            try:
                paraphrased_texts = ast.literal_eval(sample["paraphrases"]) # "['paraphrased text 1', 'paraphrased text 2', ...]" -> ['paraphrased text 1', 'paraphrased text 2', ...]
            except:
                print("Error")
                continue
            paraphrase_encoding = bert_tokenizer(paraphrased_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                outputs = bert_model(**paraphrase_encoding, return_dict=True, output_hidden_states=True)
            paraphrase_embeddings = outputs.hidden_states[-1][:, 0, :]
            original_text = sample["text"]
            original_encoding = bert_tokenizer(original_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                outputs = bert_model(**original_encoding, return_dict=True, output_hidden_states=True)
            original_embeddings = outputs.hidden_states[-1][:, 0, :]
            # use euclidean distance 
            distances = torch.nn.functional.pairwise_distance(original_embeddings, paraphrase_embeddings, p=2)
            most_similar = torch.argmin(distances)
            most_dissimilar = torch.argmax(distances)
            # or, use cosine similarity (make negligible difference)
            # similarities = torch.nn.functional.cosine_similarity(original_embedding, paraphrase_embeddings, dim=-1)
            # most_similar = torch.argmax(similarities)
            # most_dissimilar = torch.argmin(similarities)
            dpo_sample = {
                "instruction": "You will be given a sentence. Please paraphrase the sentence. ",
                "input": original_text,
                "chosen": paraphrased_texts[most_dissimilar],
                "rejected": paraphrased_texts[most_similar]
            }
            paraphrase_dpo.append(dpo_sample)
    random.shuffle(paraphrase_dpo, random.seed(42))
    with open(f"dataset-base/my-data/paraphrases/paraphrase_dpo_{SFT_SUFFIX}_{DPO_SUFFIX}.json", "w") as f:
        json.dump(paraphrase_dpo, f, indent=2)

process_dpo()
