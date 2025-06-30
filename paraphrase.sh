for dataset in "anli" "ChemProt" "cola" "mnli" "mpqa" "mrpc" "rct" "rte" "sst2" "symptoms" "yelp" "subj"; do
    CUDA_VISIBLE_DEVICES="0" python paraphrase.py --dataset_name "$dataset" --llm_stage "sft" --save_sim # ablation: wo_dpo, wo_dpo|ds
    CUDA_VISIBLE_DEVICES="0" python paraphrase.py --dataset_name "$dataset" --llm_stage "dpo" --save_sim --save_par # method, ablation: wo_ds
    CUDA_VISIBLE_DEVICES="0" python paraphrase.py --dataset_name "$dataset" --llm_stage "dpo" --wo_coreset # ablation
    CUDA_VISIBLE_DEVICES="0" python paraphrase.py --dataset_name "$dataset" --llm_stage "dpo" --wo_selective # ablation
    CUDA_VISIBLE_DEVICES="0" python paraphrase.py --dataset_name "$dataset" --llm_stage "dpo" --base_model "qwen" # qwen 
done
