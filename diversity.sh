for dataset in "anli" "ChemProt" "symptoms" "mnli" "yelp" "sst2" "cola" "mrpc" "rte" "mpqa" "subj" "rct"; do
    CUDA_VISIBLE_DEVICES="1" python diversity.py --dataset_name "$dataset" --do_ori --do_cst --do_aug --do_abl --do_bsl --do_var # example
done
