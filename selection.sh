for dataset in "anli" "ChemProt" "cola" "mnli" "mpqa" "mrpc" "rct" "rte" "sst2" "symptoms" "yelp" "subj"; do
    CUDA_VISIBLE_DEVICES="0" python selection.py --dataset_name "$dataset" --seed 42 --reduce --reduction_size 1200
    CUDA_VISIBLE_DEVICES="0" python train.py --dataset_name "$dataset" --do_ori --do_cst # find coreset
done
