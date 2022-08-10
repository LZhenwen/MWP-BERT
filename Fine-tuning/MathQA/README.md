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
