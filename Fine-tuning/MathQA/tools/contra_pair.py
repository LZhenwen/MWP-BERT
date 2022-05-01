# make contra pair, pos sample and neg sample
# create data/pairs/xxx.json

import json
from src.pre_data import *
from src.expressions_transfer import *
import tqdm
import random


def score(node):
    if not isinstance(node, list):
        return 1
    return score(node[1][0]) + score(node[1][1]) + 1

def tree_normalize(node):
    # pdb.set_trace()
    if (not isinstance(node, list)):
        return node
    score0 = score(node[1][0])
    score1 = score(node[1][1])

    if (score0 > score1):
        
        tmp = node[1][1]
        node[1][1] = node[1][0]
        node[1][0] = tmp
    
    tree_normalize(node[1][0])
    tree_normalize(node[1][1])

    return node

def maketree(forword):
    # construct tree
    valid_op = ['+', '-', '*', '/']
    root = None
    now = root
    for word in forword:
        if word in valid_op:
            node = [word, [None, None], None]
            if now == None:
                root = node
                now = root
            else:
                if now[1][0] == None:
                    now[1][0] = node
                    node[2] = now
                    now = node
                else:
                    now[1][1] = node
                    node[2] = now
                    now = node
        else:
            if now == None and root == None:
                root = word
            elif now != None:
                if now[1][0] == None:
                    now[1][0] = word
                else:
                    now[1][1] = word
                    while(now != None and now[1][1] != None):
                        now = now[2]
    if now != None:
        print("error", forword)
    # root = tree_normalize(root)
    return root

def add_order(tree, order):
    if isinstance(tree, list):
        tree.append(order)
        new_order = add_order(tree[1][0], order + 1)
        new_order = add_order(tree[1][1], new_order)
        return new_order
    else:
        return order + 1

def comp(t1, t2):
    if not isinstance(t1, list) and not isinstance(t2, list):
        return True
    if isinstance(t1, list) and isinstance(t2, list) and t1[0] == t2[0]:
        return comp(t1[1][0], t2[1][0]) and comp(t1[1][1], t2[1][1])
    return False

def is_sub_tree(t, s):
    if not isinstance(t, list):
        return -1
    if not isinstance(s, list):
        return -1
    if comp(t, s):
        return s[3]
    else:
        x1 = is_sub_tree(t, s[1][0])
        if x1 != -1:
            return x1
        x2 = is_sub_tree(t, s[1][1])
        return x2

def common_tree(t, s):
    if not isinstance(t, list):
        return (-1, -1)
    if score(t) <= 3:
        return (-1, -1)
    x = is_sub_tree(t, s)
    if x != -1:
        return (t[3], x)

    x = common_tree(t[1][0], s)
    if x != (-1, -1):
        return x
    x = common_tree(t[1][1], s)
    return x

def main():
    data1 = json.load(open("data/Math_23K_mbert_token_train.json"))["pairs"]
    data2 = json.load(open("data/MathQA_mbert_token_train.json"))["pairs"]

    pairs = []
    data_pairs = []
    cnt_0 = 0

    tree_ls1 = [maketree(from_infix_to_prefix(_['expression'])) for _ in data1]
    tree_ls2 = [maketree(from_infix_to_prefix(_['expression'])) for _ in data2]
    for _ in tree_ls1:
        add_order(_, 0)
    for _ in tree_ls2:
        add_order(_, 0)

    for i, t1 in enumerate(tqdm.tqdm(tree_ls1)):
        for j, t2 in enumerate(tree_ls2):
            if j <= i:
                continue
            if comp(t1, t2):
                pairs.append([i,j])

    for i, t1 in enumerate(tqdm.tqdm(tree_ls1)):
        for j, t2 in enumerate(tree_ls2):

            x = is_sub_tree(t1, t2)
            if  x != -1:
                pairs.append([i,j,0,x])
            if x != 0:
                x = is_sub_tree(t2, t1)
                if x != -1:
                    pairs.append([j,i,x,0])

    json.dump(pairs, open("data/pairs/Math_23K-MathQA.json", "w"), indent=4)

def func1(neg_idx, idx, ops, n_num):
    if (ops[idx] == ops[neg_idx] and n_num[idx] != n_num[neg_idx]):
        return True
    return False

def func2(neg_idx, idx, ops, n_num):
    if (ops[idx] != ops[neg_idx] and n_num[idx] == n_num[neg_idx]):
        return True
    return False

def RetTrue(*args):
    return True

def sample(path, sample_num, add_nopair=False, neg_samp=False, contra_sub_tree_pos=False, neg_samp_func=RetTrue):
    data1 = json.load(open("data/Math_23K_mbert_token_train.json"))["pairs"]
    data2 = json.load(open("data/MathQA_mbert_token_train.json"))["pairs"]

    data = data1 + data2
    pairs = json.load(open(path))

    tree_ls1 = [maketree(from_infix_to_prefix(_['expression'])) for _ in data1]
    tree_ls2 = [maketree(from_infix_to_prefix(_['expression'])) for _ in data2]
    trees = tree_ls1 + tree_ls2
    ops = [tree[0] for tree in trees]
    exprs = [from_infix_to_postfix(_['expression']) for _ in data]
    n_num = [len([_ for _ in expr if _ not in ['+', '-', '*', '/']]) for expr in exprs]

    for pair in tqdm.tqdm(pairs):
        if pair[2] == 0:
            pair[1] += len(data1)
        else:
            pair[0] += len(data1)

    # for pair in tqdm.tqdm(pairs):
    #     pair[1] += len(data1)

    pairs_dict = dict()
    for pair in tqdm.tqdm(pairs):
        if pair[0] not in pairs_dict:
            pairs_dict[pair[0]] = []
        pairs_dict[pair[0]].append(pair[1])
    random.shuffle(pairs)

    # sample
    d = dict()
    new_pairs = []
    pair_pos = []
    for pair in tqdm.tqdm(pairs):
        if pair[0] not in d:
            d[pair[0]] = 0
        if pair[1] not in d:
            d[pair[1]] = 0
        if d[pair[0]] >= sample_num or d[pair[1]] >= sample_num:
            continue
        d[pair[0]] += 1
        d[pair[1]] += 1
        new_pair = [pair[0], pair[1]]
        new_pairs.append(new_pair)
        if contra_sub_tree_pos:
            pair_pos.append([pair[2], pair[3]])
    
    # d = [_ for _ in d if d[_] > 0]
    # print(len(d), len(data1) + len(data2))

    if add_nopair:
        tot_len = len(data1) + len(data2)
        for id in range(tot_len):
            if id not in d:
                new_pairs.append([id, id]) # if not match, match it with itself
    if neg_samp:
        for new_pair in tqdm.tqdm(new_pairs):
            for i in range(5):
                cnt = 0
                while(True):
                    if new_pair[0] < len(data1):
                        neg = random.randint(0, len(data2)-1) + len(data1)
                        if neg not in pairs_dict[new_pair[0]] and neg_samp_func(neg, new_pair[0], ops, n_num):
                            new_pair.append(neg)
                            break
                    else:
                        neg = random.randint(0, len(data1)-1)
                        if neg not in pairs_dict[new_pair[0]] and neg_samp_func(neg, new_pair[0], ops, n_num):
                            new_pair.append(neg)
                            break    
                    cnt += 1
                    if cnt > 1000:
                        new_pair.append(neg)
                        break
    
    if contra_sub_tree_pos:
        json.dump({"pairs": new_pairs, "pos": pair_pos}, open("data/pairs/Math_23K-MathQA-sample.json", "w"), indent=4)
    else:
        json.dump(new_pairs, open("data/pairs/Math_23K-MathQA-sample.json", "w"), indent=4)


if __name__ == '__main__':
    main() # find all pos pairs
    # total pairs are too much, need to sample
    sample('data/pairs/Math_23K-MathQA.json', 4, neg_samp=True, contra_sub_tree_pos=True, neg_samp_func=func1)
