# DoAug

## Overview

This is the code repository for "Diversity-oriented Data Augmentation with Large Language Models" (ACL 2025)

TL;DR: We fine-tune an LLM to generate diverse paraphrases for textual data augmentation. 

The paper is available at [arXiv](https://arxiv.org/abs/2502.11671).

## Run Commands

1. prepare paraphrase dataset
```shell
bash prepare_paraphrase_dataset.sh
```
2. use LLaMA-Factory to train a paraphrase model
```shell
cd LLaMA-Factory
bash LLaMA-Factory/llamafactory.sh
cd ../
```
3. select the best coreset
```shell
bash selection.sh
```
4. paraphrase generation
```shell
bash paraphrase.sh
```
5. task performance evaluation
```shell
bash train.sh
```
6. diversity and affinity evaluation
```shell
bash diversity.sh
```

## Citation

```bibtex
@article{wang2025diversity,
  title={Diversity-Oriented Data Augmentation with Large Language Models},
  author={Wang, Zaitian and Zhang, Jinghan and Zhang, Xinhao and Liu, Kunpeng and Wang, Pengfei and Zhou, Yuanchun},
  journal={arXiv preprint arXiv:2502.11671},
  year={2025}
}
```