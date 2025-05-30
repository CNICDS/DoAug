import os
import json
    
def save_dataset_as_jsonl(dataset, dataset_name, split_name):
    with open(os.path.join(dataset_name, f"{split_name}.jsonl"), "w") as f:
        for sample in dataset:
            f.write(json.dumps(sample) + "\n")

def get_coreset_key(dataset_name):
    if dataset_name in ["anli", "sst2", "mrpc", "mpqa", "rct", "multi"]:
        return "variance"
    if dataset_name in ["mnli", "yelp", "cola", "rte"]:
        return "ccs_aum"
    if dataset_name in ["ChemProt", "subj"]:
        return "entropy"
    if dataset_name in ["symptoms"]:
        return "el2n"
    return "variance"

def get_num_labels(DATASET_NAME):
    if DATASET_NAME in ["anli", "mnli", "multi"]:
        return 3
    elif DATASET_NAME == "ChemProt":
        return 13
    elif DATASET_NAME == "symptoms":
        return 25
    elif DATASET_NAME == "yelp":
        return 5
    elif DATASET_NAME == "rct":
        return 5
    else:
        return 2
    
def get_test_file(DATASET_NAME):
    if DATASET_NAME in ["mnli", "sst2", "cola", "mrpc", "qqp", "qnli", "rte", "wnli"]:
        return "validation.jsonl"
    else:
        return "test.jsonl"

def get_data_files(CORESET_KEY=None, do_cst=False, do_aug=False, do_abl=False, do_bsl=False, do_var=False, test_file="test.jsonl", use_clean=False):
    data_files = {
        # test
        'test': test_file,
        # original
        'reduced_800': 'reduced_800.jsonl', # randomly reduced from 1200        
        # special group
        # 'train': 'train.jsonl', # full dataset comparison
        # 'reduced_1200': 'reduced_1200.jsonl' # low-resource comparison (randomly sampled from full)
    }
    data_files_cst = { # coreset
        # ablation group 0
        ## w/o augmentation -> select the best coreset key
        'el2n_top_2_3': 'coreset_el2n_top_2_3.jsonl',
        'entropy_top_2_3': 'coreset_entropy_top_2_3.jsonl',
        'variance_top_2_3': 'coreset_variance_top_2_3.jsonl',
        'ccs_aum_top_2_3': 'coreset_ccs_aum_top_2_3.jsonl',
    }
    data_files_aug = { # augmentation
        # method
        'dpo_d': f'aug_doaug_dpo50k_sel_{CORESET_KEY}_dissimilar.jsonl', # DoAug
    }
    data_files_abl = { # ablation
        # ablation group 1
        'wo_coreset_dpo_d': f'aug_doaug_dpo50k_wocst_{CORESET_KEY}_dissimilar.jsonl', # w/o coreset (CORESET_KEY not used here, just don't want to add more rules when coding)
        'wo_selective_dpo_d': f'aug_doaug_dpo50k_wosel_{CORESET_KEY}_dissimilar.jsonl', # w/o selective
        # ablation group 2
        'sft_s': f'aug_doaug_sft100k_sel_{CORESET_KEY}_similar.jsonl', # w/o DPO|DS
        'sft_d': f'aug_doaug_sft100k_sel_{CORESET_KEY}_dissimilar.jsonl', # w/o DPO
        'dpo_s': f'aug_doaug_dpo50k_sel_{CORESET_KEY}_similar.jsonl', # w/o DS
    }
    suffix_clean = "_clean" if use_clean else ""
    data_files_bsl = { # baseline
        # baseline group 1
        'ocr': f'baseline_ocr{suffix_clean}.jsonl',
        'typing': f'baseline_keyboard{suffix_clean}.jsonl',
        'eda': 'baseline_eda.jsonl',
        'aeda': 'baseline_aeda.jsonl',
        # baseline group 2
        'bt': 'baseline_bt.jsonl',
        'maskpred': 'baseline_unmask.jsonl',
        # baseline group 3
        'auggpt': 'baseline_auggpt.jsonl',
        'gramma': 'baseline_grammar.jsonl',
        'spell': 'baseline_spell.jsonl',
        # baseline group 4
        'chain': 'baseline_chain.jsonl',
        'hint': 'baseline_hint.jsonl',
        'taboo': 'baseline_taboo.jsonl',
    }
    data_files_var = { # variant
        # variant group
        '10k': f'aug_doaug_dpo10k_sel_{CORESET_KEY}_dissimilar.jsonl',
        '20k': f'aug_doaug_dpo20k_sel_{CORESET_KEY}_dissimilar.jsonl',
        '50k': f'aug_doaug_dpo50k_sel_{CORESET_KEY}_dissimilar.jsonl',
        '100k': f'aug_doaug_dpo100k_sel_{CORESET_KEY}_dissimilar.jsonl',
        't1.2': f'aug_doaug_sft100k_sel_{CORESET_KEY}_dissimilar_t1.2.jsonl',
        'prompt': f'aug_doaug_sft100k_sel_{CORESET_KEY}_dissimilar_prompt.jsonl',
        'qwen': f'aug_doaug_dpo50k_sel_{CORESET_KEY}_dissimilar_qwen.jsonl',
    }
    if do_cst:
        data_files.update(data_files_cst)
    if do_aug:
        data_files.update(data_files_aug)
    if do_abl:
        data_files.update(data_files_abl)
    if do_bsl:
        data_files.update(data_files_bsl)
    if do_var:
        data_files.update(data_files_var)
    return data_files

def get_splits():
    SPLITS = [ # should keep the order
        # original
        'reduced_800', # randomly reduced from 1200        
        # method
        'dpo_d', # DoAug
        # ablation group 0
        ## w/o augmentation -> select the best coreset key
        'el2n_top_2_3',
        'entropy_top_2_3',
        'variance_top_2_3',
        'ccs_aum_top_2_3',
        # ablation group 1
        'wo_coreset_dpo_d', # w/o coreset
        'wo_selective_dpo_d', # w/o selective
        # ablation group 2
        'sft_s', # w/o DPO|DS
        'sft_d', # w/o DPO
        'dpo_s', # w/o DS
        # baseline group 1
        'ocr',
        'typing',
        'eda',
        'aeda',
        # baseline group 2
        'bt',
        'maskpred',
        # baseline group 3
        'auggpt',
        'gramma',
        'spell',
        # baseline group 4
        'chain',
        'hint',
        'taboo',
        # special group
        # 'train', # full dataset comparison
        # 'reduced_1200' # low-resource comparison (randomly sampled from full)
        # variant group
        '10k',
        '20k',
        '50k',
        '100k',
        't1.2',
        'prompt',
        'qwen'
    ]
    return SPLITS

def get_used_splits(do_ori=False, do_cst=False, do_aug=False, do_abl=False, do_bsl=False, do_var=False, CORESET_KEY=None):
    SPLITS = []
    ORI_SPLITS = [
        # original
        'reduced_800', # randomly reduced from 1200  
        # special group
        # 'train', # full dataset comparison
        # 'reduced_1200' # low-resource comparison (randomly sampled from full)
    ]
    CST_SPLITS = [
        # ablation group 0
        ## w/o augmentation -> select the best coreset key
        'el2n_top_2_3',
        'entropy_top_2_3',
        'variance_top_2_3',
        'ccs_aum_top_2_3',
    ]
    AUG_SPLITS = [
        # method
        'dpo_d', # DoAug
    ]
    ABL_SPLITS = [
        # ablation group 1
        'wo_coreset_dpo_d', # w/o coreset
        'wo_selective_dpo_d', # w/o selective
        # ablation group 2
        'sft_s', # w/o DPO|DS
        'sft_d', # w/o DPO
        'dpo_s', # w/o DS
    ]
    BSL_SPLITS = [
        # baseline group 1
        'ocr',
        'typing',
        'eda',
        'aeda',
        # # baseline group 2
        'bt',
        'maskpred',
        # # baseline group 3
        'auggpt',
        'gramma',
        'spell',
        # # baseline group 4
        'chain',
        'hint',
        'taboo',
    ]
    VAR_SPLITS = [
        # variant group
        '10k',
        '20k',
        '50k',
        '100k',
        't1.2',
        'prompt',
        'qwen'
    ]
    if do_ori:
        SPLITS += ORI_SPLITS
    if do_cst:
        if CORESET_KEY:
            SPLITS += [f"{CORESET_KEY}_top_2_3"]
        else:
            SPLITS += CST_SPLITS
    if do_aug:
        SPLITS += AUG_SPLITS
    if do_abl:
        SPLITS += ABL_SPLITS
    if do_bsl:
        SPLITS += BSL_SPLITS
    if do_var:
        SPLITS += VAR_SPLITS
    return SPLITS

