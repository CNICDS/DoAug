cp ../dataset-base/my-data/paraphrases/* data/
mkdir llm-base/doaug

CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS="16" llamafactory-cli train  examples/train_lora/llama3_lora_sft.yaml
CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS="16" llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS="16" llamafactory-cli train  examples/train_lora/llama3_lora_dpo.yaml
CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS="16" llamafactory-cli export examples/merge_lora/llama3_lora_dpo.yaml
cp -r models/llama_paraphraser_sft_100k_100k ../llm-base/doaug
cp -r models/llama_paraphraser_dpo_100k_50k ../llm-base/doaug
