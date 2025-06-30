import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from transformers import pipeline
from copy import deepcopy
from argparse import ArgumentParser
from utils import *

args = ArgumentParser()

args.add_argument("--dataset_name", type=str, default="anli")
args.add_argument("--llm_stage", type=str, default="sft", choices=["sft", "dpo"])
args.add_argument("--temperature", type=float, default=1.0)
args.add_argument("--save_sim", action='store_true', default=False)
args.add_argument("--save_par", action='store_true', default=False)
args.add_argument("--wo_coreset", action='store_true', default=False)
args.add_argument("--wo_selective", action='store_true', default=False)
args.add_argument("--dpo_size", type=str, default="50k", choices=["10k", "20k", "50k", "100k"])
args.add_argument("--base_model", type=str, default="llama", choices=["llama", "qwen"])
args = args.parse_args()
print(args)

if args.dataset_name in ["multi"]:
    BERT_PATH = "llm-base/google-bert/bert-base-multilingual-cased"
else:
    BERT_PATH = "llm-base/google-bert/bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert_model = AutoModelForMaskedLM.from_pretrained(BERT_PATH).to("cuda")
bert_model.to("cuda")
bert_model.eval()

SFT_SUFFIX = "100k"
DPO_SUFFIX = args.dpo_size

DATASET_NAME = args.dataset_name
LLM_STAGE = args.llm_stage
if LLM_STAGE == "sft":
    DPO_SUFFIX = "100k"

BASE_MODEL = args.base_model

model_name = f"llm-base/doaug/{BASE_MODEL}_paraphraser_{LLM_STAGE}_{SFT_SUFFIX}_{DPO_SUFFIX}"
print(f"Generating with {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
quantization = torch.float16
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=quantization)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, torch_dtype=quantization, temperature=args.temperature)

def paraphrase(text):
    multilingual_prompt = "Note that the sentence maybe either english, french, german, spanish, italian, portuguese, or hindi. " if DATASET_NAME in ["multi"] else ""
    message = [
        {
            "content": "You will be given a sentence. Please paraphrase the sentence. " + multilingual_prompt,
            "role": "system"
        },
        {
            "content": text,
            "role": "user"
        }
    ]
    PARAPHRASE_NUM = 5
    output = pipe(message, max_new_tokens=1000, num_return_sequences=PARAPHRASE_NUM, do_sample=True)
    return [each["generated_text"][-1]['content'] for each in output]

def paraphrase_select(text):
    paraphrased_texts = paraphrase(text) # [5]
    paraphrase_encoding = bert_tokenizer(paraphrased_texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = bert_model(**paraphrase_encoding, return_dict=True, output_hidden_states=True)
    paraphrase_embeddings = outputs.hidden_states[-1][:, 0, :]
    original_encoding = bert_tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = bert_model(**original_encoding, return_dict=True, output_hidden_states=True)
    original_embedding = outputs.hidden_states[-1][:, 0, :]
    distance = torch.nn.functional.pairwise_distance(paraphrase_embeddings, original_embedding)
    most_dissimilar = torch.argmax(distance)
    most_similar = torch.argmin(distance)
    # or, use cosine similarity (should make negligible difference)
    # cosine_similarity = torch.nn.functional.cosine_similarity(paraphrase_embeddings, original_embedding, dim=1)
    # most_dissimilar = torch.argmin(cosine_similarity)
    # most_similar = torch.argmax(cosine_similarity)
    return paraphrased_texts[most_dissimilar], paraphrased_texts[most_similar]

print(f"Processing {DATASET_NAME}")
dataset_name = f"dataset-base/my-data/{DATASET_NAME}"
CORESET_KEY = get_coreset_key(DATASET_NAME)
dataset = load_dataset(dataset_name, data_files={
    'selected_top': f'coreset_{CORESET_KEY}_top_1_3.jsonl', # doaug
    'selected_mid': f'coreset_{CORESET_KEY}_mid_1_3.jsonl', # doaug
    'reduced_800': "reduced_800.jsonl", # abl
    'selected_top_800': f"coreset_{CORESET_KEY}_top_2_3.jsonl", # abl
})
if args.wo_coreset: # no coreset at all, 800 randomly from 1200 and split into 400/400
    seed_dataset = dataset["reduced_800"].shuffle(seed=42).select(range(0, 400))
    noneseed_dataset = dataset["reduced_800"].shuffle(seed=42).select(range(400, len(dataset["reduced_800"])))
if args.wo_selective: # 800 coreset from 1200, randomly split into 400/400
    seed_dataset = dataset["selected_top_800"].shuffle(seed=42).select(range(0, 400))
    noneseed_dataset = dataset["selected_top_800"].shuffle(seed=42).select(range(400, len(dataset["selected_top_800"])))
else: 
    seed_dataset = dataset["selected_top"]
    noneseed_dataset = dataset["selected_mid"]

similar_augmented_dataset = deepcopy(seed_dataset) # for ablation
dissimilar_augmented_dataset = deepcopy(seed_dataset) # outcome
dissimilar_paraphrase_dataset = deepcopy(seed_dataset) # for validation
dissimilar_paraphrased_samples = []


for sample_idx, sample in enumerate(tqdm(seed_dataset)):
    if DATASET_NAME == "csqa":
        question = sample["question"]
        choices = sample["choices"]
        answerKey = sample["answerKey"]
        raw_text = question
    elif DATASET_NAME == "codah":
        question = sample["question_propmt"]
        choices = sample["candidate_answers"]
        answerKey = sample["correct_answer_idx"]
        raw_text = question
    else:
        raw_text = sample["text"]
        label = sample["label"]
    # augment text
    if DATASET_NAME in ['anli', 'mnli']:
        # split the sentence by 'Premise: ' and 'Hypothesis: '
        prm = raw_text.split('Premise: ')[1].split('Hypothesis: ')[0]
        hyp = raw_text.split('Hypothesis: ')[1]
        prm_augmented_dis, prm_augmented_sim = paraphrase_select(prm)
        hyp_augmented_dis, hyp_augmented_sim = paraphrase_select(hyp)
        augmented_text_dis = f"Premise: {prm_augmented_dis}. Hypothesis: {hyp_augmented_dis}."
        augmented_text_sim = f"Premise: {prm_augmented_sim}. Hypothesis: {hyp_augmented_sim}."
    elif DATASET_NAME in ['mrpc', 'rte']:
        # split the sentence by 'sentence 1: ' and 'sentence 2: '
        s1 = raw_text.split('Sentence 1: ')[1].split('Sentence 2: ')[0]
        s2 = raw_text.split('Sentence 2: ')[1]
        s1_augmented_dis, s1_augmented_sim = paraphrase_select(s1)
        s2_augmented_dis, s2_augmented_sim = paraphrase_select(s2)
        augmented_text_dis = f"Sentence 1: {s1_augmented_dis}. Sentence 2: {s2_augmented_dis}."
        augmented_text_sim = f"Sentence 1: {s1_augmented_sim}. Sentence 2: {s2_augmented_sim}."
    else:
        augmented_text_dis, augmented_text_sim = paraphrase_select(raw_text)
    # combine dataset
    if DATASET_NAME == "csqa":
        dissimilar_augmented_dataset = dissimilar_augmented_dataset.add_item(
            {"question": augmented_text_dis, "choices": choices, "answerKey": answerKey})
        similar_augmented_dataset = similar_augmented_dataset.add_item(
            {"question": augmented_text_sim, "choices": choices, "answerKey": answerKey})
    elif DATASET_NAME == "codah":
        dissimilar_augmented_dataset = dissimilar_augmented_dataset.add_item(
            {"question": augmented_text_dis, "candidate_answers": choices, "correct_answer_idx": answerKey})
        similar_augmented_dataset = similar_augmented_dataset.add_item(
            {"question": augmented_text_sim, "candidate_answers": choices, "correct_answer_idx": answerKey})
    else:
        dissimilar_augmented_dataset = dissimilar_augmented_dataset.add_item(
            {"text": augmented_text_dis, "label": label})
        similar_augmented_dataset = similar_augmented_dataset.add_item(
            {"text": augmented_text_sim, "label": label})
    dissimilar_paraphrased_samples.append(augmented_text_dis)
for sample in noneseed_dataset:
    dissimilar_augmented_dataset = dissimilar_augmented_dataset.add_item(sample)
    similar_augmented_dataset = similar_augmented_dataset.add_item(sample)
dissimilar_paraphrase_dataset = dissimilar_paraphrase_dataset.add_column("paraphrased_text", dissimilar_paraphrased_samples)

DATASET_SUFFIX = SFT_SUFFIX if LLM_STAGE == 'sft' else DPO_SUFFIX
ABLATION_SUFFIX = "wocst" if args.wo_coreset else "wosel" if args.wo_selective else "sel"
MODEL_SUFFIX = f"_{args.base_model}" if args.base_model != "llama" else ""
save_dataset_as_jsonl(dissimilar_augmented_dataset, dataset_name, f"aug_doaug_{LLM_STAGE}{DATASET_SUFFIX}_{ABLATION_SUFFIX}_{CORESET_KEY}_dissimilar{MODEL_SUFFIX}")
if args.save_sim:
    save_dataset_as_jsonl(similar_augmented_dataset, dataset_name, f"aug_doaug_{LLM_STAGE}{DATASET_SUFFIX}_{ABLATION_SUFFIX}_{CORESET_KEY}_similar{MODEL_SUFFIX}")
if args.save_par:
    save_dataset_as_jsonl(dissimilar_paraphrase_dataset, dataset_name, f"para_doaug_{LLM_STAGE}{DATASET_SUFFIX}_{ABLATION_SUFFIX}_{CORESET_KEY}_dissimilar{MODEL_SUFFIX}")
