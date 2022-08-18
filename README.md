# MWP-BERT
NAACL 2022 Findings Paper: MWP-BERT: Numeracy-Augmented Pre-training for Math Word Problem Solving

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mwp-bert-a-strong-baseline-for-math-word/math-word-problem-solving-on-mathqa)](https://paperswithcode.com/sota/math-word-problem-solving-on-mathqa?p=mwp-bert-a-strong-baseline-for-math-word)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mwp-bert-a-strong-baseline-for-math-word/math-word-problem-solving-on-math23k)](https://paperswithcode.com/sota/math-word-problem-solving-on-math23k?p=mwp-bert-a-strong-baseline-for-math-word)

## Pre-training on Chinese dataset.

**1. How to pre-train a MWP-BERT model:**
```
python all_pretrain.py
```

**1. How to pre-train a MWP-RoBERTa model:**
```
python all_pretrain_roberta.py
```

## Pre-training on English dataset.

**1. How to pre-train a MWP-BERT model:**
```
python en_pretrain.py
```

# MWP-BERT on Chinese Dataset

## Network Training

**1. How to train the network on Math23k dataset only:**
```
python math23k.py
```

**2. How to train the network on Math23k dataset and Ape-clean jointly:**
```
python math23k_wape.py
```
## Weight Loading

To load the pre-trained MWP-BERT model or other pre-trained models in Huggingface, there are two lines of code need changing:

**1. src/models.py, Line 232:**
```
self.bert_rnn = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
```
Load the model from your desired path.

**2. src/models.py, Line 803/903/1039:**
```
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
```
Load the tokenizer from your backbone model.

## MWP-BERT weights

Please find at https://drive.google.com/drive/folders/1QC7b6dnUSbHLJQHJQNwecPNiQQoBFu8T?usp=sharing.

# MWP-BERT on English Dataset

We build our implementation based on the code from https://github.com/zwx980624/mwp-cl, thanks for their contribution!

## Network Training

**1. How to train the network on MathQA dataset:**
```
run mathqa.sh
```

## Weight Loading

To load the pre-trained MWP-BERT model, just change:

**mathqa.sh, Line 5:**
```
--bert_pretrain_path pretrained_models/bert-base-uncased \
```

Load the pre-trained model from your desired path.

## MWP-BERT weights

Please find at https://drive.google.com/drive/folders/1QC7b6dnUSbHLJQHJQNwecPNiQQoBFu8T?usp=sharing.

## Citation

```
@inproceedings{liang2022mwp,
  title={MWP-BERT: Numeracy-Augmented Pre-training for Math Word Problem Solving},
  author={Liang, Zhenwen and Zhang, Jipeng and Wang, Lei and Qin, Wei and Lan, Yunshi and Shao, Jie and Zhang, Xiangliang},
  booktitle={Findings of NAACL 2022},
  pages={997--1009},
  year={2022}
}
```
