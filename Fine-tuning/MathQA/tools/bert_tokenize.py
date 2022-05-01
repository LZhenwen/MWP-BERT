# coding: utf-8
# data preprocess: bert tokenize ; create data/xxx_token_xxx.json
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import json


def main():
    data = load_raw_data("data/Math_23K_train.json")

    pairs, generate_nums, copy_nums = transfer_num(data)
    bert_tokenizer = BertTokenizer.from_pretrained('../pretrained_models/bert-base-multilingual-uncased')

    # bert_tokenizer.add_tokens(['[num]'])
    bert_tokenizer.add_special_tokens({"additional_special_tokens":["[num]"]})
    count = 0
    new_items = []
    for item in pairs:
        old_token = item[0]
        sent = ""
        for token in old_token:
            if (token == 'NUM'):
                sent += " " + "[num]"
            else:
                sent += " " + token
        sent = "[CLS]" + sent + "[SEP]"
        new_token = bert_tokenizer.tokenize(sent)
        new_num_pos = [] # use bert to tokenize，numpos changed
        for i, token in enumerate(new_token):
            if (token == '[num]' or token == '[NUM]'):
                new_num_pos.append(i)
        if len(new_num_pos) != len(item[2]):
            print("new num error")
            print("old:", old_token)
            print("new:", new_token)
            count += 1
        new_item = {
            "tokens": new_token,
            "expression": item[1],
            "nums": item[2],
            "num_pos": new_num_pos
        }
        new_items.append(new_item)
    print(count)
    print(123)
    json.dump({"pairs": new_items, "generate_nums": generate_nums, "copy_nums": copy_nums}, open("data/Math_23K_mbert_token_train.json", "w"), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()