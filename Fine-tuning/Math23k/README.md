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
