import time
import torch.optim
import json
import torch
import random
import copy
import re
from src.pre_data import *
from copy import deepcopy
import torch.nn.functional as f
import torch.nn as nn
from transformers import BertModel, BertForMaskedLM, BertTokenizer


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    indices = indices.cuda()
    masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0)

def mask_tokens(inputs, bert_tokenizer: BertTokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()

    padding_mask = labels.clone()

    inputs[padding_mask == -1] = 0

    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [bert_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).bool(), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix)
    labels[(masked_indices != 1.0) ] = -100  # We only compute loss on masked tokens

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).cuda()
    inputs[(indices_replaced == 1.0) & (labels == -100)] = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.mask_token)

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).cuda()
    random_words = torch.randint(len(bert_tokenizer), labels.shape, dtype=torch.long).cuda()
    inputs[(indices_random == 1.0) & (indices_replaced == 1.0) & (labels == -100)] = random_words[(indices_random == 1.0) & (indices_replaced == 1.0) & (labels == -100)]

    return inputs, labels


def get_train_test_fold_23k(ori_path,prefix,data,pairs,group):

    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item,pair in zip(data, pairs):
        pair = list(pair)
        pair = tuple(pair)
        if pair[3] in train_id:
            train_fold.append(pair)
        if pair[3] in test_id:
            test_fold.append(pair)
    return train_fold, test_fold, valid_fold

def get_train_test_fold_all(ori_path,prefix,data,pairs,group,ape_path,ape_id,ape_test_id):
    id_list = open(ape_id, 'r').read().split()
    id_list = [str(i) for i in id_list]
    test_id_list = open(ape_test_id, 'r').read().split()
    test_id_list = [str(i) for i in test_id_list]
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item,pair in zip(data, pairs):
        pair = list(pair)
        pair = tuple(pair)
        train_fold.append(pair)
    return train_fold, test_fold, valid_fold

def get_train_test_fold_all_pretrain(ori_path,prefix,data,pairs,group,ape_path,ape_id,ape_test_id):
    id_list = open(ape_id, 'r').read().split()
    id_list = [str(i) for i in id_list]
    test_id_list = open(ape_test_id, 'r').read().split()
    test_id_list = [str(i) for i in test_id_list]
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item,pair in zip(data, pairs):
        pair = list(pair)
        pair = tuple(pair)
        if pair[-1] == 'ape_train':
            train_fold.append(pair)
        elif pair[-1] == 'ape_test':
            test_fold.append(pair)
        else:
            train_fold.append(pair)
    return train_fold, test_fold, valid_fold

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num

def transfer_num_pretrain(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    #count = 0
    for d in data:
        #count += 1
        #if count == 100:
        #    break
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        out_seq = from_infix_to_prefix(out_seq)
        candi = []
        for token in out_seq:
            if 'N' in token and out_seq.count(token) == 1:
                candi.append(token)
        #print(deepcopy(out_seq))
        try:
            if len(out_seq) > 1 or 'N' in str(out_seq):
                tree_list = pre_process_tree(deepcopy(out_seq))
                length_predict_list, token_predict_list = generate_pretraining(tree_list, candi)
            else:
                length_predict_list, token_predict_list = [], []
        except:
            continue
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos, length_predict_list, token_predict_list))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums

class Tree_node():
    def __init__(self, key, parent = None, left_child = None, right_child = None):
        self.key = key
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child

def pre_process_tree(a):
    op = ['+', '-', '*', '/', '^']
    stack = []
    result = []
    while a:
        key = a.pop(0)
        if stack == [] and key in op:
            stack.append(Tree_node(key))
        else:
            current = stack.pop()
            if current.left_child == None:
                current.left_child = key
                stack.append(current)
            else:
                current.right_child = key
                result.append(current)
            if key in op:
                stack.append(Tree_node(key, current))
            else:
                result.append(Tree_node(key, current))
    return result

def prepare_data_pretraining(pairs_trained, pairs_tested, pairs_tested_ape, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    test_pairs_ape = []
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    for pair in pairs_trained:
        num_stack = []
        for idx in range(len(pair[0])):
            if pair[0][idx] == 'NUM':
                pair[0][idx] = 'n'
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])
        #if num_stack != []:
        #    print(num_stack)
        #    print('!!!')
        #inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt",add_special_tokens=False)
        inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

        num_pos = []
        for idx,i in enumerate(inputs['input_ids'].squeeze()):
            if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                num_pos.append(idx)
        
        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        #print(len(output_cell))
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        if output_lang.word2index["UNK"] in output_cell:
            continue
        if len(input_cell) > 100 or len(output_cell) > 20:
            continue
        if len(output_cell) <= 1:
            continue
        if max(pair[3]) >= inputs['input_ids'].squeeze().size(0):
            continue
        train_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                            pair[2], num_pos, num_stack, inputs, pair[4], pair[5]))
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    return input_lang, output_lang, train_pairs


def prepare_train_batch(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    bert_input_batches = []
    #label_batches = []
    dist_batches = []
    operator_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        #label_length = []
        for _, i, _, j, _, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
            #label_length.append(len(label))
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        #label_length_max = max(label_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        bert_input_batch = []
        dist_batch = []
        operator_batch = []
        #label_batch = []
        for i, li, j, lj, num, num_pos, num_stack, bert_input, token_dist, token_operator in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            bert_input_batch.append(bert_input)
            #label_batch.append(label + [-1] * (label_length_max - len(label)))
            dist_batch.append(token_dist)
            operator_batch.append(token_operator)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        bert_input_batches.append(bert_input_batch)
        #label_batches.append(label_batch)
        dist_batches.append(dist_batch)
        operator_batches.append(operator_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, bert_input_batches, dist_batches, operator_batches


class PreTrainBert(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(PreTrainBert, self).__init__()

        self.bert_model = BertForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.dist_loss = torch.nn.MSELoss()
        #self.fc = nn.Linear(hidden_size, 2)
        self.fc_dist = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size * 2,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=1))
        self.fc_op = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size * 2,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=5))

    def forward(self, bert_input, num_pos, dist, operator):
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_input])
        input_ids = []
        attention_mask = []
        for i in bert_input:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().cuda()
        input_masked, label = mask_tokens(input_ids, self.tokenizer)
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        batch_size = len(dist)
        output = self.bert_model(input_masked, attention_mask=attention_mask, output_hidden_states=True, labels = label)
        bert_output = output.hidden_states[-1].transpose(0,1)
        loss_lmm = output.loss
        x = get_all_number_encoder_outputs(bert_output, num_pos, batch_size, num_size, self.hidden_size)
        
        loss_dist = []
        loss_op = []
        for idx in range(batch_size):
            for dist_idx in range(len(dist[idx])):
                #print(dist[idx])
                target = int(dist[idx][dist_idx][2])
                #print(target)
                idx1, idx2 = int(dist[idx][dist_idx][0].strip('N')), int(dist[idx][dist_idx][1].strip('N'))
                dist_feature = torch.cat([x[idx][idx1], x[idx][idx2]], dim = -1)
                #print(int(dist[idx][dist_idx][2]))
                loss_dist.append(self.dist_loss(self.fc_dist(dist_feature), torch.FloatTensor([target]).cuda()))
            for op_idx in range(len(operator[idx])):
                #print(int(operator[idx][op_idx][2]))
                target = int(operator[idx][op_idx][2])
                #print(target)
                idx1, idx2 = int(operator[idx][op_idx][0].strip('N')), int(operator[idx][op_idx][1].strip('N'))
                op_feature = torch.cat([x[idx][idx1], x[idx][idx2]], dim = -1)
                #print(self.fc_op(op_feature).shape)
                #print(torch.LongTensor(int(operator[idx][op_idx][2])))
                #print(torch.tensor([target]).cuda().unsqueeze(0))
                loss_op.append(self.loss_function(self.fc_op(op_feature).unsqueeze(0), torch.tensor([target]).cuda()))
        if len(loss_dist):
            loss_dist = sum(loss_dist) / len(loss_dist)
        else:
            loss_dist = 0
        if len(loss_op):
            loss_op = sum(loss_op) / len(loss_op)
        else:
            loss_op = 0
    
        return loss_dist + loss_op + loss_lmm

def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res

def generate_pretraining(result, candi):
    output_list = []
    token_predict_list = []
    op = ['+', '-', '*', '/', '^']
    for idx_1 in range(len(result)):
        if result[idx_1].key in op and result[idx_1].left_child in candi and result[idx_1].right_child in candi:
            token_predict_list.append([result[idx_1].left_child, result[idx_1].right_child, op.index(result[idx_1].key)])
        for idx_2 in range(idx_1 + 1, len(result)):
            if result[idx_1].key in candi and result[idx_2].key in candi and result[idx_1].key != result[idx_2].key:
                joint_parents = []
                if result[idx_1].parent == result[idx_2].parent:
                    output_list.append([result[idx_1].key, result[idx_2].key, 2])
                    continue
                else:
                    current_1 = result[idx_1]
                    current_2 = result[idx_2]
                    path_len_1 = 1
                    path_len_2 = 1
                    while 1:
                        joint_parents.append([current_1.parent, path_len_1])
                        joint_parents.append([current_2.parent, path_len_2])
                        if current_1.parent:
                            current_1 = current_1.parent
                            path_len_1 += 1
                        if current_2.parent:
                            current_2 = current_2.parent
                            path_len_2 += 1
                        break_flag = False
                        for i in joint_parents:
                            if current_1.parent == i[0]:
                                output_list.append([result[idx_1].key, result[idx_2].key, path_len_1 + i[1]])
                                break_flag = True
                                break
                            elif current_2.parent == i[0]:
                                output_list.append([result[idx_1].key, result[idx_2].key, path_len_2 + i[1]])
                                break_flag = True
                                break
                            elif current_1.parent == current_2.parent:
                                output_list.append([result[idx_1].key, result[idx_2].key, path_len_1 + path_len_2])
                                break_flag = True
                                break
                        if break_flag:
                            break
    return output_list, token_predict_list


def transfer_num_pretrain_weak(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    #count = 0
    for d in data:
        #count += 1
        #if count == 100:
        #    break
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]
        ans = d['ans']
        if '/' in ans or '%' in ans or '.' in ans:
            ans_label = 1
        else:
            ans_label = 0
        fractions = re.findall("\d+\(\d+\/\d+\)", ans)
        if len(fractions):
            ans = ans.replace("(", "+(")
        try:
            ans_value = float(to_nums(ans))
        except:
            continue

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        out_seq = from_infix_to_prefix(out_seq)
        candi = []
        for token in out_seq:
            if 'N' in token and out_seq.count(token) == 1:
                candi.append(token)
        #print(deepcopy(out_seq))

        quantity_labels = []
        type_labels = []
        try:
            for n in nums:
                if ans_value >= to_nums(n):
                    quantity_labels.append(1)
                else:
                    quantity_labels.append(0)
                if '/' in n or '%' in n or '.' in n:
                    num_label = 1
                else:
                    num_label = 0
                type_labels.append(ans_label ^ num_label)
        except:
            continue
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos, ans_label, quantity_labels, type_labels))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums

def to_nums(string):
    string = str(string)
    string = string.replace('%','/100')
    return eval(string)

def prepare_data_pretraining_weak(pairs_trained, pairs_tested, pairs_tested_ape, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    test_pairs_ape = []
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    for pair in pairs_trained:
        try:
            num_stack = []
            for idx in range(len(pair[0])):
                if pair[0][idx] == 'NUM':
                    pair[0][idx] = 'n'
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])
            #if num_stack != []:
            #    print(num_stack)
            #    print('!!!')
            #inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt",add_special_tokens=False)
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

            num_pos = []
            for idx,i in enumerate(inputs['input_ids'].squeeze()):
                if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                    num_pos.append(idx)
            
            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            #print(len(output_cell))
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            train_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                pair[2], num_pos, num_stack, inputs, pair[4], pair[5], pair[6]))
        except:
            continue
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    return input_lang, output_lang, train_pairs

def prepare_train_batch_weak(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    bert_input_batches = []
    #label_batches = []
    answer_label_batches = []
    quantity_label_batches = []
    type_label_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        #label_length = []
        for _, i, _, j, _, _, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
            #label_length.append(len(label))
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        #label_length_max = max(label_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        bert_input_batch = []
        answer_label_batch = []
        quantity_label_batch = []
        type_label_batch = []
        for i, li, j, lj, num, num_pos, num_stack, bert_input, ans_label, quantity_label, type_label in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            bert_input_batch.append(bert_input)
            #label_batch.append(label + [-1] * (label_length_max - len(label)))
            answer_label_batch.append(ans_label)
            quantity_label_batch.append(quantity_label)
            type_label_batch.append(type_label)
        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        bert_input_batches.append(bert_input_batch)
        #label_batches.append(label_batch)
        answer_label_batches.append(answer_label_batch)
        quantity_label_batches.append(quantity_label_batch)
        type_label_batches.append(type_label_batch)
    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, bert_input_batches, answer_label_batches, quantity_label_batches, type_label_batches


class PreTrainBert_weak(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(PreTrainBert_weak, self).__init__()

        self.bert_model = BertForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        #self.dist_loss = torch.nn.MSELoss()
        #self.fc = nn.Linear(hidden_size, 2)
        self.fc_ans_type = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_quantity_rela = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_type_rela = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))

    def forward(self, bert_input, num_pos, answer_type, quantity_relation, type_relation):
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_input])
        input_ids = []
        attention_mask = []
        for i in bert_input:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().cuda()
        input_masked, label = mask_tokens(input_ids, self.tokenizer)
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        batch_size = len(answer_type)
        output = self.bert_model(input_masked, attention_mask=attention_mask, output_hidden_states=True, labels = label)
        bert_output = output.hidden_states[-1].transpose(0,1)
        problem_output = bert_output.mean(0).squeeze()
        loss_1 = self.loss_function(self.fc_ans_type(problem_output), torch.tensor(answer_type).cuda())
        #loss_lmm = output.loss
        x = get_all_number_encoder_outputs(bert_output, num_pos, batch_size, num_size, self.hidden_size)
        max_len = x.shape[1]
        for idx in range(len(quantity_relation)):
            quantity_relation[idx] += [-1 for _ in range(max_len - len(quantity_relation[idx]))]
            type_relation[idx] += [-1 for _ in range(max_len - len(type_relation[idx]))]

        loss_2 = self.loss_function(self.fc_quantity_rela(x).view(-1, 2), torch.tensor(quantity_relation).cuda().view(-1))
        loss_3 = self.loss_function(self.fc_type_rela(x).view(-1, 2), torch.tensor(type_relation).cuda().view(-1))
    
        return loss_1 + loss_2 + loss_3




def transfer_num_pretrain_self(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    #count = 0
    for d in data:
        #count += 1
        #if count == 100:
        #    break
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        out_seq = from_infix_to_prefix(out_seq)
        candi = []
        for token in out_seq:
            if 'N' in token and out_seq.count(token) == 1:
                candi.append(token)
        #print(deepcopy(out_seq))

        num_num = len(nums)
        num_label = []
        try:
            for n in nums:
                if '/' in n or '%' in n or '.' in n:
                    num_label.append(1)
                else:
                    num_label.append(0)
        except:
            continue
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos, num_num, num_label))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums

def prepare_data_pretraining_self(pairs_trained, pairs_tested, pairs_tested_ape, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    test_pairs_ape = []
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    for pair in pairs_trained:
        try:
            num_stack = []
            for idx in range(len(pair[0])):
                if pair[0][idx] == 'NUM':
                    pair[0][idx] = 'n'
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])
            #if num_stack != []:
            #    print(num_stack)
            #    print('!!!')
            #inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt",add_special_tokens=False)
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

            num_pos = []
            for idx,i in enumerate(inputs['input_ids'].squeeze()):
                if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                    num_pos.append(idx)
            
            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            #print(len(output_cell))
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if output_lang.word2index["UNK"] in output_cell:
                continue
            if len(input_cell) > 100 or len(output_cell) > 20:
                continue
            if len(output_cell) <= 1:
                continue
            if max(pair[3]) >= inputs['input_ids'].squeeze().size(0):
                continue
            train_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                pair[2], num_pos, num_stack, inputs, pair[4], pair[5]))
        except:
            continue
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    return input_lang, output_lang, train_pairs

def prepare_train_batch_self(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    bert_input_batches = []
    #label_batches = []
    num_num_batches = []
    num_label_batches = []
    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        #label_length = []
        for _, i, _, j, _, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
            #label_length.append(len(label))
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        #label_length_max = max(label_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        bert_input_batch = []
        num_num_batch = []
        num_label_batch = []

        for i, li, j, lj, num, num_pos, num_stack, bert_input, num_num, num_label in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            bert_input_batch.append(bert_input)
            #label_batch.append(label + [-1] * (label_length_max - len(label)))
            num_num_batch.append(num_num)
            num_label_batch.append(num_label)

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        bert_input_batches.append(bert_input_batch)
        #label_batches.append(label_batch)
        num_num_batches.append(num_num_batch)
        num_label_batches.append(num_label_batch)

    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, bert_input_batches, num_num_batches, num_label_batches


class PreTrainBert_self(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(PreTrainBert_self, self).__init__()

        self.bert_model = BertForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.dist_loss = torch.nn.MSELoss()
        #self.fc = nn.Linear(hidden_size, 2)
        self.fc_num_num = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=1))
        self.fc_num_label = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))


    def forward(self, bert_input, num_pos, num_num, num_label):
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_input])
        input_ids = []
        attention_mask = []
        for i in bert_input:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().cuda()
        input_masked, label = mask_tokens(input_ids, self.tokenizer)
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        batch_size = len(num_num)
        output = self.bert_model(input_masked, attention_mask=attention_mask, output_hidden_states=True, labels = label)
        bert_output = output.hidden_states[-1].transpose(0,1)
        problem_output = bert_output.mean(0).squeeze()
        loss_1 = self.dist_loss(self.fc_num_num(problem_output), torch.FloatTensor(num_num).cuda().unsqueeze(1))
        loss_lmm = output.loss
        x = get_all_number_encoder_outputs(bert_output, num_pos, batch_size, num_size, self.hidden_size)
        max_len = x.shape[1]
        for idx in range(len(num_label)):
            num_label[idx] += [-1 for _ in range(max_len - len(num_label[idx]))]

        loss_2 = self.loss_function(self.fc_num_label(x).view(-1, 2), torch.LongTensor(num_label).cuda().view(-1))
    
        return loss_1 + loss_2 + loss_lmm

def transfer_num_pretrain_all(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    #count = 0
    for d in data:
        #count += 1
        #if count == 100:
        #    break
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]
        ans = d['ans']
        if '/' in ans or '%' in ans or '.' in ans:
            ans_label = 1
        else:
            ans_label = 0
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        try:
            ans_value = float(to_nums(ans))
        except:
            continue

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        out_seq = from_infix_to_prefix(out_seq)
        candi = []
        for token in out_seq:
            if 'N' in token and out_seq.count(token) == 1:
                candi.append(token)
        #print(deepcopy(out_seq))

        num_num = len(nums)
        num_label = []
        try:
            for n in nums:
                if '/' in n or '%' in n or '.' in n:
                    num_label.append(1)
                else:
                    num_label.append(0)
        except:
            continue

        quantity_labels = []
        type_labels = []
        try:
            for n in nums:
                if ans_value >= to_nums(n):
                    quantity_labels.append(1)
                else:
                    quantity_labels.append(0)
                if '/' in n or '%' in n or '.' in n:
                    num_label_weak = 1
                else:
                    num_label_weak = 0
                type_labels.append(ans_label ^ num_label_weak)
        except:
            continue

        #print(deepcopy(out_seq))
        try:
            if len(out_seq) > 1 or 'N' in str(out_seq):
                tree_list = pre_process_tree(deepcopy(out_seq))
                length_predict_list, token_predict_list = generate_pretraining(tree_list, candi)
            else:
                length_predict_list, token_predict_list = [], []
        except:
            continue
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos, num_num, num_label, ans_label, quantity_labels, type_labels, length_predict_list, token_predict_list, d["type"]))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums

def prepare_data_pretraining_all(pairs_trained, pairs_tested, pairs_tested_ape, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    test_pairs_ape = []
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    for pair in pairs_trained:
        try:
            num_stack = []
            for idx in range(len(pair[0])):
                if pair[0][idx] == 'NUM':
                    pair[0][idx] = 'n'
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])
            #if num_stack != []:
            #    print(num_stack)
            #    print('!!!')
            #inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt",add_special_tokens=False)
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

            num_pos = []
            for idx,i in enumerate(inputs['input_ids'].squeeze()):
                if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                    num_pos.append(idx)
            
            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            #print(len(output_cell))
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if output_lang.word2index["UNK"] in output_cell:
                continue
            if len(input_cell) > 100 or len(output_cell) > 20:
                continue
            if len(output_cell) <= 1:
                continue
            if max(pair[3]) >= inputs['input_ids'].squeeze().size(0):
                continue
            train_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                pair[2], num_pos, num_stack, inputs, pair[4], pair[5], pair[6], pair[7], pair[8], pair[9], pair[10]))
        except:
            continue
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        try:
            num_stack = []
            for idx in range(len(pair[0])):
                if pair[0][idx] == 'NUM':
                    pair[0][idx] = 'n'
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])
            #if num_stack != []:
            #    print(num_stack)
            #    print('!!!')
            #inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt",add_special_tokens=False)
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

            num_pos = []
            for idx,i in enumerate(inputs['input_ids'].squeeze()):
                if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                    num_pos.append(idx)
            
            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            #print(len(output_cell))
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if output_lang.word2index["UNK"] in output_cell:
                continue
            if len(input_cell) > 100 or len(output_cell) > 20:
                continue
            if len(output_cell) <= 1:
                continue
            if max(pair[3]) >= inputs['input_ids'].squeeze().size(0):
                continue
            test_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                pair[2], num_pos, num_stack, inputs, pair[4], pair[5], pair[6], pair[7], pair[8], pair[9], pair[10]))
        except:
            continue    
    print('Number of testing data %d' % (len(test_pairs)))

    return input_lang, output_lang, train_pairs, test_pairs

def prepare_data_pretraining_all_rbt(pairs_trained, pairs_tested, pairs_tested_ape, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    test_pairs_ape = []
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    for pair in pairs_trained:
        try:
            num_stack = []
            for idx in range(len(pair[0])):
                if pair[0][idx] == 'NUM':
                    pair[0][idx] = 'n'
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])
            #if num_stack != []:
            #    print(num_stack)
            #    print('!!!')
            #inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt",add_special_tokens=False)
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

            num_pos = []
            for idx,i in enumerate(inputs['input_ids'].squeeze()):
                if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                    num_pos.append(idx)
            
            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            #print(len(output_cell))
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if output_lang.word2index["UNK"] in output_cell:
                continue
            if len(input_cell) > 100 or len(output_cell) > 20:
                continue
            if len(output_cell) <= 1:
                continue
            if max(pair[3]) >= inputs['input_ids'].squeeze().size(0):
                continue
            train_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                pair[2], num_pos, num_stack, inputs, pair[4], pair[5], pair[6], pair[7], pair[8], pair[9], pair[10]))
        except:
            continue
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        try:
            num_stack = []
            for idx in range(len(pair[0])):
                if pair[0][idx] == 'NUM':
                    pair[0][idx] = 'n'
            for word in pair[1]:
                temp_num = []
                flag_not = True
                if word not in output_lang.index2word:
                    flag_not = False
                    for i, j in enumerate(pair[2]):
                        if j == word:
                            temp_num.append(i)

                if not flag_not and len(temp_num) != 0:
                    num_stack.append(temp_num)
                if not flag_not and len(temp_num) == 0:
                    num_stack.append([_ for _ in range(len(pair[2]))])
            #if num_stack != []:
            #    print(num_stack)
            #    print('!!!')
            #inputs = tokenizer(pair[0], is_split_into_words=True,return_tensors="pt",add_special_tokens=False)
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

            num_pos = []
            for idx,i in enumerate(inputs['input_ids'].squeeze()):
                if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                    num_pos.append(idx)
            
            num_stack.reverse()
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            #print(len(output_cell))
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))
            if output_lang.word2index["UNK"] in output_cell:
                continue
            if len(input_cell) > 100 or len(output_cell) > 20:
                continue
            if len(output_cell) <= 1:
                continue
            if max(pair[3]) >= inputs['input_ids'].squeeze().size(0):
                continue
            test_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                pair[2], num_pos, num_stack, inputs, pair[4], pair[5], pair[6], pair[7], pair[8], pair[9], pair[10]))
        except:
            continue    
    print('Number of testing data %d' % (len(test_pairs)))

    return input_lang, output_lang, train_pairs, test_pairs


def prepare_train_batch_all(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    nums_batches = []
    batches = []
    input_batches = []
    output_batches = []
    num_stack_batches = []  # save the num stack which
    num_pos_batches = []
    num_size_batches = []
    bert_input_batches = []
    #label_batches = []
    num_num_batches = []
    num_label_batches = []
    answer_label_batches = []
    quantity_label_batches = []
    type_label_batches = []
    dist_batches = []
    operator_batches = []

    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        #label_length = []
        for _, i, _, j, _, _, _, _, _, _, _, _, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
            #label_length.append(len(label))
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        #label_length_max = max(label_length)
        input_batch = []
        output_batch = []
        num_batch = []
        num_stack_batch = []
        num_pos_batch = []
        num_size_batch = []
        bert_input_batch = []
        num_num_batch = []
        num_label_batch = []
        answer_label_batch = []
        quantity_label_batch = []
        type_label_batch = []
        dist_batch = []
        operator_batch = []

        for i, li, j, lj, num, num_pos, num_stack, bert_input, num_num, num_label, ans_label, quantity_label, type_label, token_dist, token_operator in batch:
            num_batch.append(len(num))
            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq(j, lj, output_len_max))
            num_stack_batch.append(num_stack)
            num_pos_batch.append(num_pos)
            num_size_batch.append(len(num_pos))
            bert_input_batch.append(bert_input)
            #label_batch.append(label + [-1] * (label_length_max - len(label)))
            num_num_batch.append(num_num)
            num_label_batch.append(num_label)
            answer_label_batch.append(ans_label)
            quantity_label_batch.append(quantity_label)
            type_label_batch.append(type_label)
            dist_batch.append(token_dist)
            operator_batch.append(token_operator)

        input_batches.append(input_batch)
        nums_batches.append(num_batch)
        output_batches.append(output_batch)
        num_stack_batches.append(num_stack_batch)
        num_pos_batches.append(num_pos_batch)
        num_size_batches.append(num_size_batch)
        bert_input_batches.append(bert_input_batch)
        #label_batches.append(label_batch)
        num_num_batches.append(num_num_batch)
        num_label_batches.append(num_label_batch)
        answer_label_batches.append(answer_label_batch)
        quantity_label_batches.append(quantity_label_batch)
        type_label_batches.append(type_label_batch)
        dist_batches.append(dist_batch)
        operator_batches.append(operator_batch)

    return input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, bert_input_batches, \
        num_num_batches, num_label_batches, answer_label_batches, quantity_label_batches, type_label_batches, dist_batches, operator_batches


class Bert_all(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(Bert_all, self).__init__()

        #self.bert_model = BertForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.bert_model = BertForMaskedLM.from_pretrained("/ibex/scratch/lianz0a/bert2tree-master/models/all_epoch_140_")
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.dist_loss = torch.nn.MSELoss()
        #self.fc = nn.Linear(hidden_size, 2)
        self.fc_num_num = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=1))
        self.fc_num_label = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_dist = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size * 2,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=1))
        self.fc_op = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size * 2,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=5))
        self.fc_ans_type = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_quantity_rela = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_type_rela = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))                                                                    

    def forward(self, bert_input, num_pos, num_num, num_label, answer_type, quantity_relation, type_relation, dist, operator):
        result = []
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_input])
        input_ids = []
        attention_mask = []
        for i in bert_input:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().cuda()
        input_masked, label = mask_tokens(input_ids, self.tokenizer)
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        batch_size = len(num_num)
        output = self.bert_model(input_masked, attention_mask=attention_mask, output_hidden_states=True, labels = label)
        bert_output = output.hidden_states[-1].transpose(0,1)
        problem_output = bert_output.mean(0).squeeze()
        loss_1 = self.dist_loss(self.fc_num_num(problem_output), torch.FloatTensor(num_num).cuda().unsqueeze(1))
        result.append(loss_1)
        loss_lmm = output.loss
        x = get_all_number_encoder_outputs(bert_output, num_pos, batch_size, num_size, self.hidden_size)
        max_len = x.shape[1]
        for idx in range(len(num_label)):
            num_label[idx] += [-1 for _ in range(max_len - len(num_label[idx]))]
        pre_loss_2 = self.fc_num_label(x).view(-1, 2)
        target_loss_2 = torch.LongTensor(num_label).cuda().view(-1)
        loss_2 = self.loss_function(pre_loss_2, target_loss_2)
        prediction = torch.argmax(pre_loss_2, 1)
        correct = (prediction == target_loss_2).sum().float()
        acc_2 = correct/(target_loss_2.shape[0] - (target_loss_2 == -1).sum().float())
        result.append(acc_2)

        loss_dist = []
        loss_op = []
        op_correct = 0
        op_sum = 0
        for idx in range(batch_size):
            for dist_idx in range(len(dist[idx])):

                target = int(dist[idx][dist_idx][2])

                idx1, idx2 = int(dist[idx][dist_idx][0].strip('N')), int(dist[idx][dist_idx][1].strip('N'))
                dist_feature = torch.cat([x[idx][idx1], x[idx][idx2]], dim = -1)

                loss_dist.append(self.dist_loss(self.fc_dist(dist_feature), torch.FloatTensor([target]).cuda()))
            for op_idx in range(len(operator[idx])):
                target = int(operator[idx][op_idx][2])

                idx1, idx2 = int(operator[idx][op_idx][0].strip('N')), int(operator[idx][op_idx][1].strip('N'))
                op_feature = torch.cat([x[idx][idx1], x[idx][idx2]], dim = -1)
                op_pre = self.fc_op(op_feature).unsqueeze(0)
                op_target = torch.tensor([target]).cuda()
                op_loss = self.loss_function(op_pre, op_target)
                
                prediction = torch.argmax(op_pre, 1)
                op_correct += (prediction == op_target).sum().float()
                op_sum += op_target.shape[0]
                loss_op.append(op_loss)
        if len(loss_dist):
            loss_dist = sum(loss_dist) / len(loss_dist)
            result.append(loss_dist)
        else:
            loss_dist = 0
            result.append(0)
        if len(loss_op):
            loss_op = sum(loss_op) / len(loss_op)
            result.append(op_correct/op_sum)
        else:
            result.append(0)
            loss_op = 0   



        pre_loss_1_weak = self.fc_ans_type(problem_output)
        target_loss_1_weak = torch.tensor(answer_type).cuda()
        loss_1_weak = self.loss_function(pre_loss_1_weak, target_loss_1_weak)
        prediction = torch.argmax(pre_loss_1_weak, 1)
        correct = (prediction == target_loss_1_weak).sum().float()
        acc_loss_1_weak = correct/target_loss_1_weak.shape[0]
        result.append(acc_loss_1_weak)        

        #loss_lmm = output.loss
        x = get_all_number_encoder_outputs(bert_output, num_pos, batch_size, num_size, self.hidden_size)
        max_len = x.shape[1]
        for idx in range(len(quantity_relation)):
            quantity_relation[idx] += [-1 for _ in range(max_len - len(quantity_relation[idx]))]
            type_relation[idx] += [-1 for _ in range(max_len - len(type_relation[idx]))]

        pre_loss_2_weak = self.fc_quantity_rela(x).view(-1, 2)
        target_loss_2_weak = torch.tensor(quantity_relation).cuda().view(-1)
        loss_2_weak = self.loss_function(pre_loss_2_weak, target_loss_2_weak)
        prediction = torch.argmax(pre_loss_2_weak, 1)
        correct = (prediction == target_loss_2_weak).sum().float()
        acc_loss_2_weak = correct/(target_loss_2_weak.shape[0] - (target_loss_2_weak == -1).sum().float())
        result.append(acc_loss_2_weak) 

        pre_loss_3_weak = self.fc_type_rela(x).view(-1, 2)
        target_loss_3_weak = torch.tensor(type_relation).cuda().view(-1)
        loss_3_weak = self.loss_function(pre_loss_3_weak, target_loss_3_weak)
        prediction = torch.argmax(pre_loss_3_weak, 1)
        correct = (prediction == target_loss_3_weak).sum().float()
        acc_loss_3_weak = correct/(target_loss_3_weak.shape[0] - (target_loss_3_weak == -1).sum().float())
        result.append(acc_loss_3_weak) 

        loss_3_weak = self.loss_function(pre_loss_3_weak, target_loss_3_weak)

        return loss_1 + loss_2 + loss_lmm + loss_dist + loss_op + loss_1_weak + loss_2_weak + loss_3_weak, result

class PreTrainBert_all(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(PreTrainBert_all, self).__init__()

        self.bert_model = BertForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.dist_loss = torch.nn.MSELoss()
        #self.fc = nn.Linear(hidden_size, 2)
        self.fc_num_num = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=1))
        self.fc_num_label = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_dist = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size * 2,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=1))
        self.fc_op = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size * 2,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=5))
        self.fc_ans_type = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_quantity_rela = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_type_rela = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))                                                                    

    def forward(self, bert_input, num_pos, num_num, num_label, answer_type, quantity_relation, type_relation, dist, operator):
        result = []
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_input])
        input_ids = []
        attention_mask = []
        for i in bert_input:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().cuda()
        input_masked, label = mask_tokens(input_ids, self.tokenizer)
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        batch_size = len(num_num)
        output = self.bert_model(input_masked, attention_mask=attention_mask, output_hidden_states=True, labels = label)
        bert_output = output.hidden_states[-1].transpose(0,1)
        problem_output = bert_output.mean(0).squeeze()
        loss_1 = self.dist_loss(self.fc_num_num(problem_output), torch.FloatTensor(num_num).cuda().unsqueeze(1))
        result.append(loss_1)
        loss_lmm = output.loss
        x = get_all_number_encoder_outputs(bert_output, num_pos, batch_size, num_size, self.hidden_size)
        max_len = x.shape[1]
        for idx in range(len(num_label)):
            num_label[idx] += [-1 for _ in range(max_len - len(num_label[idx]))]
        pre_loss_2 = self.fc_num_label(x).view(-1, 2)
        target_loss_2 = torch.LongTensor(num_label).cuda().view(-1)
        loss_2 = self.loss_function(pre_loss_2, target_loss_2)
        prediction = torch.argmax(pre_loss_2, 1)
        correct = (prediction == target_loss_2).sum().float()
        acc_2 = correct/(target_loss_2.shape[0] - (target_loss_2 == -1).sum().float())
        result.append(acc_2)

        loss_dist = []
        loss_op = []
        op_correct = 0
        op_sum = 0
        for idx in range(batch_size):
            for dist_idx in range(len(dist[idx])):

                target = int(dist[idx][dist_idx][2])

                idx1, idx2 = int(dist[idx][dist_idx][0].strip('N')), int(dist[idx][dist_idx][1].strip('N'))
                dist_feature = torch.cat([x[idx][idx1], x[idx][idx2]], dim = -1)

                loss_dist.append(self.dist_loss(self.fc_dist(dist_feature), torch.FloatTensor([target]).cuda()))
            for op_idx in range(len(operator[idx])):
                target = int(operator[idx][op_idx][2])

                idx1, idx2 = int(operator[idx][op_idx][0].strip('N')), int(operator[idx][op_idx][1].strip('N'))
                op_feature = torch.cat([x[idx][idx1], x[idx][idx2]], dim = -1)
                op_pre = self.fc_op(op_feature).unsqueeze(0)
                op_target = torch.tensor([target]).cuda()
                op_loss = self.loss_function(op_pre, op_target)
                
                prediction = torch.argmax(op_pre, 1)
                op_correct += (prediction == op_target).sum().float()
                op_sum += op_target.shape[0]
                loss_op.append(op_loss)
        if len(loss_dist):
            loss_dist = sum(loss_dist) / len(loss_dist)
            result.append(loss_dist)
        else:
            loss_dist = 0
            result.append(0)
        if len(loss_op):
            loss_op = sum(loss_op) / len(loss_op)
            result.append(op_correct/op_sum)
        else:
            result.append(0)
            loss_op = 0   



        pre_loss_1_weak = self.fc_ans_type(problem_output)
        target_loss_1_weak = torch.tensor(answer_type).cuda()
        loss_1_weak = self.loss_function(pre_loss_1_weak, target_loss_1_weak)
        prediction = torch.argmax(pre_loss_1_weak, 1)
        correct = (prediction == target_loss_1_weak).sum().float()
        acc_loss_1_weak = correct/target_loss_1_weak.shape[0]
        result.append(acc_loss_1_weak)        

        #loss_lmm = output.loss
        x = get_all_number_encoder_outputs(bert_output, num_pos, batch_size, num_size, self.hidden_size)
        max_len = x.shape[1]
        for idx in range(len(quantity_relation)):
            quantity_relation[idx] += [-1 for _ in range(max_len - len(quantity_relation[idx]))]
            type_relation[idx] += [-1 for _ in range(max_len - len(type_relation[idx]))]

        pre_loss_2_weak = self.fc_quantity_rela(x).view(-1, 2)
        target_loss_2_weak = torch.tensor(quantity_relation).cuda().view(-1)
        loss_2_weak = self.loss_function(pre_loss_2_weak, target_loss_2_weak)
        prediction = torch.argmax(pre_loss_2_weak, 1)
        correct = (prediction == target_loss_2_weak).sum().float()
        acc_loss_2_weak = correct/(target_loss_2_weak.shape[0] - (target_loss_2_weak == -1).sum().float())
        result.append(acc_loss_2_weak) 

        pre_loss_3_weak = self.fc_type_rela(x).view(-1, 2)
        target_loss_3_weak = torch.tensor(type_relation).cuda().view(-1)
        loss_3_weak = self.loss_function(pre_loss_3_weak, target_loss_3_weak)
        prediction = torch.argmax(pre_loss_3_weak, 1)
        correct = (prediction == target_loss_3_weak).sum().float()
        acc_loss_3_weak = correct/(target_loss_3_weak.shape[0] - (target_loss_3_weak == -1).sum().float())
        result.append(acc_loss_3_weak) 

        loss_3_weak = self.loss_function(pre_loss_3_weak, target_loss_3_weak)

        return loss_1 + loss_2 + loss_lmm + loss_dist + loss_op + loss_1_weak + loss_2_weak + loss_3_weak, result




class PreTrainBert_all_rbt(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(PreTrainBert_all_rbt, self).__init__()

        #self.bert_model = BertForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.bert_model = BertForMaskedLM.from_pretrained("/ibex/scratch/lianz0a/bert2tree-master/models/rbt_all_epoch_140")
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.dist_loss = torch.nn.MSELoss()
        #self.fc = nn.Linear(hidden_size, 2)
        self.fc_num_num = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=1))
        self.fc_num_label = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_dist = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size * 2,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=1))
        self.fc_op = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size * 2,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=5))
        self.fc_ans_type = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_quantity_rela = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))
        self.fc_type_rela = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=hidden_size),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_size,
                                                                    out_features=2))                                                                    

    def forward(self, bert_input, num_pos, num_num, num_label, answer_type, quantity_relation, type_relation, dist, operator):
        result = []
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_input])
        input_ids = []
        attention_mask = []
        for i in bert_input:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().cuda()
        input_masked, label = mask_tokens(input_ids, self.tokenizer)
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        batch_size = len(num_num)
        output = self.bert_model(input_masked, attention_mask=attention_mask, output_hidden_states=True, labels = label)
        bert_output = output.hidden_states[-1].transpose(0,1)
        problem_output = bert_output.mean(0).squeeze()
        loss_1 = self.dist_loss(self.fc_num_num(problem_output), torch.FloatTensor(num_num).cuda().unsqueeze(1))
        result.append(loss_1)
        loss_lmm = output.loss
        x = get_all_number_encoder_outputs(bert_output, num_pos, batch_size, num_size, self.hidden_size)
        max_len = x.shape[1]
        for idx in range(len(num_label)):
            num_label[idx] += [-1 for _ in range(max_len - len(num_label[idx]))]
        pre_loss_2 = self.fc_num_label(x).view(-1, 2)
        target_loss_2 = torch.LongTensor(num_label).cuda().view(-1)
        loss_2 = self.loss_function(pre_loss_2, target_loss_2)
        prediction = torch.argmax(pre_loss_2, 1)
        correct = (prediction == target_loss_2).sum().float()
        acc_2 = correct/(target_loss_2.shape[0] - (target_loss_2 == -1).sum().float())
        result.append(acc_2)

        loss_dist = []
        loss_op = []
        op_correct = 0
        op_sum = 0
        for idx in range(batch_size):
            for dist_idx in range(len(dist[idx])):

                target = int(dist[idx][dist_idx][2])

                idx1, idx2 = int(dist[idx][dist_idx][0].strip('N')), int(dist[idx][dist_idx][1].strip('N'))
                dist_feature = torch.cat([x[idx][idx1], x[idx][idx2]], dim = -1)

                loss_dist.append(self.dist_loss(self.fc_dist(dist_feature), torch.FloatTensor([target]).cuda()))
            for op_idx in range(len(operator[idx])):
                target = int(operator[idx][op_idx][2])

                idx1, idx2 = int(operator[idx][op_idx][0].strip('N')), int(operator[idx][op_idx][1].strip('N'))
                op_feature = torch.cat([x[idx][idx1], x[idx][idx2]], dim = -1)
                op_pre = self.fc_op(op_feature).unsqueeze(0)
                op_target = torch.tensor([target]).cuda()
                op_loss = self.loss_function(op_pre, op_target)
                
                prediction = torch.argmax(op_pre, 1)
                op_correct += (prediction == op_target).sum().float()
                op_sum += op_target.shape[0]
                loss_op.append(op_loss)
        if len(loss_dist):
            loss_dist = sum(loss_dist) / len(loss_dist)
            result.append(loss_dist)
        else:
            loss_dist = 0
            result.append(0)
        if len(loss_op):
            loss_op = sum(loss_op) / len(loss_op)
            result.append(op_correct/op_sum)
        else:
            result.append(0)
            loss_op = 0   



        pre_loss_1_weak = self.fc_ans_type(problem_output)
        target_loss_1_weak = torch.tensor(answer_type).cuda()
        loss_1_weak = self.loss_function(pre_loss_1_weak, target_loss_1_weak)
        prediction = torch.argmax(pre_loss_1_weak, 1)
        correct = (prediction == target_loss_1_weak).sum().float()
        acc_loss_1_weak = correct/target_loss_1_weak.shape[0]
        result.append(acc_loss_1_weak)        

        #loss_lmm = output.loss
        x = get_all_number_encoder_outputs(bert_output, num_pos, batch_size, num_size, self.hidden_size)
        max_len = x.shape[1]
        for idx in range(len(quantity_relation)):
            quantity_relation[idx] += [-1 for _ in range(max_len - len(quantity_relation[idx]))]
            type_relation[idx] += [-1 for _ in range(max_len - len(type_relation[idx]))]

        pre_loss_2_weak = self.fc_quantity_rela(x).view(-1, 2)
        target_loss_2_weak = torch.tensor(quantity_relation).cuda().view(-1)
        loss_2_weak = self.loss_function(pre_loss_2_weak, target_loss_2_weak)
        prediction = torch.argmax(pre_loss_2_weak, 1)
        correct = (prediction == target_loss_2_weak).sum().float()
        acc_loss_2_weak = correct/(target_loss_2_weak.shape[0] - (target_loss_2_weak == -1).sum().float())
        result.append(acc_loss_2_weak) 

        pre_loss_3_weak = self.fc_type_rela(x).view(-1, 2)
        target_loss_3_weak = torch.tensor(type_relation).cuda().view(-1)
        loss_3_weak = self.loss_function(pre_loss_3_weak, target_loss_3_weak)
        prediction = torch.argmax(pre_loss_3_weak, 1)
        correct = (prediction == target_loss_3_weak).sum().float()
        acc_loss_3_weak = correct/(target_loss_3_weak.shape[0] - (target_loss_3_weak == -1).sum().float())
        result.append(acc_loss_3_weak) 

        loss_3_weak = self.loss_function(pre_loss_3_weak, target_loss_3_weak)

        return loss_1 + loss_2 + loss_lmm + loss_dist + loss_op + loss_1_weak + loss_2_weak + loss_3_weak, result


def transfer_num_pretrain_23k(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    count = 0
    for d in data:
        count += 1
        #if count == 100:
        #    break
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]
        ans = d['ans']

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []


        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        out_seq = from_infix_to_prefix(out_seq)
        random.seed(count)
        negtive_seq = deepcopy(out_seq)
        if random.random()<0.5 and len(pairs) > 10:
            while negtive_seq == out_seq:
                negtive_seq = random.choice(pairs)[1]
        else:
            negtive_seq = perturb(out_seq)

        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, negtive_seq, str(d["id"])))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


def transfer_num_pretrain_tagging(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    #count = 0
    for d in data:
        #count += 1
        #if count == 100:
        #    break
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"].replace(' ','')
        equations = equations[2:]
        #ans = d['ans']

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []


        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res
        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        label = []
        for ii in range(len(nums)):
            try:
                idx = out_seq.index('N' + str(ii))
                if idx == 0:
                    label.append(0)
                else:
                    if out_seq[idx - 1] == '+':
                        label.append(0)
                    else:
                        label.append(1)
            except:
                label.append(2)
        out_seq = from_infix_to_prefix(out_seq)

        pairs.append((input_seq, out_seq, nums, num_pos, label))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums

def prepare_data_pretraining_23k(pairs_trained, pairs_tested, pairs_tested_ape, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    test_pairs_ape = []
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
            output_lang.add_sen_to_vocab(pair[2])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
            output_lang.add_sen_to_vocab(pair[2])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:

        for idx in range(len(pair[0])):
            if pair[0][idx] == 'NUM':
                pair[0][idx] = 'n'
        inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        neg_output = indexes_from_sentence(output_lang, pair[2], tree)
        #print(len(output_cell))
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        if output_lang.word2index["UNK"] in output_cell:
            continue
        if len(input_cell) > 100 or len(output_cell) > 20:
            continue
        if len(output_cell) <= 1:
            continue

        train_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell), neg_output, len(neg_output),
                                inputs))

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:

        for idx in range(len(pair[0])):
            if pair[0][idx] == 'NUM':
                pair[0][idx] = 'n'
        inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)

        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)
        neg_output = indexes_from_sentence(output_lang, pair[2], tree)
        #print(len(output_cell))
        # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
        #                     pair[2], pair[3], num_stack, pair[4]))
        if output_lang.word2index["UNK"] in output_cell:
            continue
        if len(input_cell) > 100 or len(output_cell) > 20:
            continue
        if len(output_cell) <= 1:
            continue

        test_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell), neg_output, len(neg_output),
                                inputs))

    print('Number of testing data %d' % (len(test_pairs)))

    return input_lang, output_lang, train_pairs, test_pairs

def prepare_train_batch_23k(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    batches = []
    input_batches = []
    output_batches = []
    bert_input_batches = []
    label_batches = []
    neg_lengthes = []

    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    #batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        neg_length = []
        #label_length = []
        for _, i, _, j, _, k, _ in batch:
            input_length.append(i)
            output_length.append(j)
            neg_length.append(k)
            #label_length.append(len(label))
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        neg_lengthes.append(neg_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length + neg_length)

        #label_length_max = max(label_length)
        input_batch = []
        output_batch = []
        label_batch = []
        bert_input_batch = []

        count = 1000
        for i, li, j, lj, k, lk, bert_input in batch:
            count += 1
            random.seed(count)
            if random.random() < 0.5:

                input_batch.append(pad_seq(i, li, input_len_max))
                output_batch.append(pad_seq_out(j, len(j), output_len_max, 30))
                bert_input_batch.append(bert_input)
                label_batch.append(1)

            else:   
                
                input_batch.append(pad_seq(i, li, input_len_max))
                output_batch.append(pad_seq_out(k, len(k), output_len_max, 30))
                bert_input_batch.append(bert_input)
                label_batch.append(0)  



        input_batches.append(input_batch)
        output_batches.append(output_batch)
        bert_input_batches.append(bert_input_batch)
        label_batches.append(label_batch)

    return input_batches, input_lengths, output_batches, output_lengths, bert_input_batches, label_batches


def prepare_train_batch_23k_test(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    batches = []
    input_batches = []
    output_batches = []
    bert_input_batches = []
    label_batches = []
    neg_lengthes = []

    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    #batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        neg_length = []
        #label_length = []
        for _, i, _, j, _, k, _ in batch:
            input_length.append(i)
            output_length.append(j)
            neg_length.append(k)
            #label_length.append(len(label))
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        neg_lengthes.append(neg_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length + neg_length)

        #label_length_max = max(label_length)
        input_batch = []
        output_batch = []
        label_batch = []
        bert_input_batch = []

        count = 2000
        for i, li, j, lj, k, lk, bert_input in batch:
            count += 1
            random.seed(count)
            if random.random() < 0.5:

                input_batch.append(pad_seq(i, li, input_len_max))
                output_batch.append(pad_seq_out(j, len(j), output_len_max, 30))
                bert_input_batch.append(bert_input)
                label_batch.append(1)

            else:   
                
                input_batch.append(pad_seq(i, li, input_len_max))
                output_batch.append(pad_seq_out(k, len(k), output_len_max, 30))
                bert_input_batch.append(bert_input)
                label_batch.append(0)  



        input_batches.append(input_batch)
        output_batches.append(output_batch)
        bert_input_batches.append(bert_input_batch)
        label_batches.append(label_batch)

    return input_batches, input_lengths, output_batches, output_lengths, bert_input_batches, label_batches

class PreTrainBert_23k(nn.Module):
    def __init__(self, hidden_size, batch_size, input_size):
        super(PreTrainBert_23k, self).__init__()

        self.bert_rnn = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        #self.bert_rnn = BertModel.from_pretrained("/home/lianz0a/294/all_epoch_140_")
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.batch_size = batch_size
        self.embedding = nn.Embedding(31, 128, padding_idx = 30)
        self.gru_pade = nn.GRU(128, hidden_size, 1, dropout=0.5)
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.5)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

                                                               

    def forward(self, bert_input, output_equation, target):
        result = []
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_input])
        input_ids = []
        attention_mask = []
        for i in bert_input:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().cuda()
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0,1)    

        problem_output = bert_output.mean(0)

        output_equation = torch.tensor(output_equation).cuda()
        embedded = self.embedding(output_equation) 
        outputs, hidden = self.gru_pade(embedded, None)
        equation_output = outputs.mean(1).squeeze()


        target = torch.tensor(target).cuda().view(-1)
        merge = torch.cat([problem_output, equation_output], dim = -1)
        pre = self.w2(self.relu(self.w1(merge)))
        loss = self.loss_function(pre, target)
        prediction = torch.argmax(pre, 1)
        correct = (prediction == target).sum().float()
        acc = correct / target.shape[0]
        
        return loss, [acc]

def pad_seq_out(seq, seq_len, max_length, padding):
    seq += [padding for _ in range(max_length - seq_len)]
    return seq

def perturb(seq):
    p_seq = deepcopy(seq)
    op = ['+','-','*','/']
    number = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5']
    while (p_seq == seq):
        for i in range(len(p_seq)):
            if random.random() < 0.3:
                if p_seq[i] in op:
                    temp = deepcopy(op)
                    temp.remove(p_seq[i])
                    p_seq[i] = random.choice(temp)
                if 'N' in p_seq[i]:
                    temp = list(set(number + [p_seq[i]]))
                    temp.remove(p_seq[i])
                    p_seq[i] = random.choice(temp)
            if p_seq != seq:
                break
    return p_seq

def pertub_problem(seq):
    for i in range(len(seq)):
        if random.random() < 0.15:
            seq[i] = '[UNK]'
    return seq


def prepare_data_pretraining_tagging(pairs_trained, pairs_tested, pairs_tested_ape, trim_min_count, generate_nums, copy_nums, tree=False):
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    test_pairs_ape = []
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    for pair in pairs_trained:
        try:
            for idx in range(len(pair[0])):
                if pair[0][idx] == 'NUM':
                    pair[0][idx] = 'n'
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
            num_pos = []
            for idx,i in enumerate(inputs['input_ids'].squeeze()):
                if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                    num_pos.append(idx)
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            #print(len(output_cell))
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))

            assert len(num_pos) == len(pair[4])
            train_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                 inputs, pair[4], num_pos))
        except:
            continue
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))

    for pair in pairs_tested:
        try:
            for idx in range(len(pair[0])):
                if pair[0][idx] == 'NUM':
                    pair[0][idx] = 'n'
            inputs = tokenizer(pair[0], is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
            num_pos = []
            for idx,i in enumerate(inputs['input_ids'].squeeze()):
                if tokenizer.convert_ids_to_tokens(int(i)) == 'n':
                    num_pos.append(idx)
            input_cell = indexes_from_sentence(input_lang, pair[0])
            output_cell = indexes_from_sentence(output_lang, pair[1], tree)
            #print(len(output_cell))
            # train_pairs.append((input_cell, len(input_cell), output_cell, len(output_cell),
            #                     pair[2], pair[3], num_stack, pair[4]))

            assert len(num_pos) == len(pair[4])
            test_pairs.append((input_cell, inputs['input_ids'].squeeze().size(0), output_cell, len(output_cell),
                                 inputs, pair[4], num_pos))
        except:
            continue   
    print('Number of testing data %d' % (len(test_pairs)))

    return input_lang, output_lang, train_pairs, test_pairs

def prepare_train_batch_tagging(pairs_to_batch, batch_size):
    pairs = copy.deepcopy(pairs_to_batch)
    random.shuffle(pairs)  # shuffle the pairs
    pos = 0
    input_lengths = []
    output_lengths = []
    batches = []
    input_batches = []
    output_batches = []
    bert_input_batches = []
    label_batches = []
    num_pos_batches = []

    while pos + batch_size < len(pairs):
        batches.append(pairs[pos:pos+batch_size])
        pos += batch_size
    #batches.append(pairs[pos:])

    for batch in batches:
        batch = sorted(batch, key=lambda tp: tp[1], reverse=True)
        input_length = []
        output_length = []
        #label_length = []
        for _, i, _, j, _, _, _ in batch:
            input_length.append(i)
            output_length.append(j)
            #label_length.append(len(label))
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        input_len_max = input_length[0]
        output_len_max = max(output_length)
        #label_length_max = max(label_length)
        input_batch = []
        output_batch = []
        label_batch = []
        bert_input_batch = []
        num_pos_batch = []


        for i, li, j, lj, bert_input, label, num_pos in batch:

            input_batch.append(pad_seq(i, li, input_len_max))
            output_batch.append(pad_seq_out(j, lj, output_len_max, 0))
            bert_input_batch.append(bert_input)
            label_batch.append(label) 
            num_pos_batch.append(num_pos)           


        input_batches.append(input_batch)
        output_batches.append(output_batch)
        bert_input_batches.append(bert_input_batch)
        label_batches.append(label_batch)
        num_pos_batches.append(num_pos_batch)

    return input_batches, input_lengths, output_batches, output_lengths, bert_input_batches, label_batches, num_pos_batches

class PreTrainBert_tagging(nn.Module):
    def __init__(self, hidden_size):
        super(PreTrainBert_tagging, self).__init__()

        #self.bert_rnn = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.bert_rnn = BertModel.from_pretrained("C://Users//InvokerLiang//Desktop//bert2tree-master//readme//all_epoch_140_")
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        #self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.LeakyReLU(0.2)

        self.fc = nn.Linear(hidden_size, 3)
        #self.w1 = nn.Linear(256, 3)

        self.dropout = nn.Dropout(0.25)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

                                                               

    def forward(self, bert_input, target, num_pos):
        result = []
        length_max = max([i['input_ids'].squeeze().size(0) for i in bert_input])
        input_ids = []
        attention_mask = []
        for i in bert_input:
            input_id = i['input_ids'].squeeze()
            mask = i['attention_mask'].squeeze()
            zeros = torch.zeros(length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
        input_ids = torch.stack(input_ids,dim = 0).long().cuda()
        attention_mask = torch.stack(attention_mask,dim = 0).long().cuda()
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0,1)  
        #q = self.wq(bert_output)  
        #k = self.wk(bert_output) 
        #v = self.wv(bert_output) 
        #bert_output, _ = scaled_dot_product_attention(q,k,v)
        #problem_output = bert_output.mean(0).unsqueeze(0)
        batch_size = bert_output.shape[1]
        loss = 0
        for idx in range(batch_size):
            number_outputs = bert_output[num_pos[idx],idx,:].view(len(num_pos[idx]), -1)
            res = self.w2(self.relu(self.w1(number_outputs)))
            number_outputs = number_outputs + res
            #r_problem_output = problem_output[:,idx,:].repeat(number_outputs.shape[0], 1)
            #score = torch.cat([number_outputs, r_problem_output], dim = -1)
            number_outputs = f.dropout(number_outputs, p = 0.25)
            pre = self.fc(number_outputs) #3 * 1

            sub_target = torch.tensor(target[idx]).cuda().view(-1)
            #print(pre.shape)
            #print(sub_target.shape)
            loss += self.loss_function(pre, sub_target)
            prediction = torch.argmax(pre, 1)
            correct = (prediction == sub_target).sum().float()
            acc = correct / (sub_target.shape[0] - (sub_target == -1).sum().float())
            result.append(acc)
        
        return loss, result

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    #计算注意力权重。
    q, k, v 必须具有匹配的前置维度。 且dq=dk
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    #虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    #但是 mask 必须能进行广播转换以便求和。

    #参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)  seq_len_k = seq_len_v
        mask: Float 张量，其形状能转换成
              (..., seq_len_q, seq_len_k)。默认为None。

    #返回值:
        #输出，注意力权重
    """
    # matmul(a,b)矩阵乘:a b的最后2个维度要能做乘法，即a的最后一个维度值==b的倒数第2个纬度值，
    # 除此之外，其他维度值必须相等或为1（为1时会广播）
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))  # 矩阵乘 =>[..., seq_len_q, seq_len_k]

    # 缩放matmul_qk
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)  # k的深度dk，或叫做depth_k
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)  # [..., seq_len_q, seq_len_k]
    # print('scaled_attention_logits:', scaled_attention_logits)

    # 将 mask 加入到缩放的张量上(重要！)
    if mask is not None:  # mask: [b, 1, 1, seq_len]
        # mask=1的位置是pad，乘以-1e9（-1*10^9）成为负无穷，经过softmax后会趋于0
        scaled_attention_logits += (mask * -1e9)

    # softmax 在最后一个轴（seq_len_k）上归一化
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)  # [..., seq_len_q, seq_len_k]
    # print('attention_weights:', attention_weights, attention_weights.dtype)

    output = torch.matmul(attention_weights, v) 
    return output, attention_weights 
