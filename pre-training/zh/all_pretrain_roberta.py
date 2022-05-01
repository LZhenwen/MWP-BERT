# coding: utf-8
from src.expressions_transfer import *
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
import json
import torch
from src.pre_training_src import *
import numpy as np

from transformers import BertModel


#torch.cuda.set_device(0)
batch_size = 128
embedding_size = 128
hidden_size = 768
n_epochs = 150
learning_rate = 1e-5
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'
ape_path = "data/ape_simple_train.json"
ape_id = "data/ape_simple_id.txt"
ape_test_id = "data/ape_simple_test_id.txt"
overlap_id = "data/overlap.txt"
test_overlap_id = "data/test_overlap.txt"

id_list = open(ape_id, 'r').read().split()
id_list = [str(i) for i in id_list]
test_id_list = open(ape_test_id, 'r').read().split()
test_id_list = [str(i) for i in test_id_list]
overlap_list = open(overlap_id, 'r').read().split()
overlap_list = [str(i) for i in overlap_list]
test_overlap_list = open(test_overlap_id, 'r').read().split()
test_overlap_list = [str(i) for i in test_overlap_list]

data =  load_raw_data("data/Math_23K.json") + raw_data_pretrain_train("data/ape_simple_train.json", overlap_list) + raw_data_pretrain_test("data/ape_simple_test.json", test_overlap_list)
group_data = read_json("data/Math_23K_processed.json")


pairs, generate_nums, copy_nums = transfer_num_pretrain_all(data)

#exit()
temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]))
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold_all_pretrain(ori_path,prefix,data,pairs,group_data,ape_path, ape_id, ape_test_id)


best_acc_fold = []

pairs_tested = test_fold
pairs_trained = train_fold

input_lang, output_lang, train_pairs, test_pairs = prepare_data_pretraining_all_rbt(pairs_trained, pairs_tested, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
pre_bert = PreTrainBert_all_rbt(hidden_size, batch_size).to(device)
#model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext").to(device)
pre_bert = torch.nn.DataParallel(pre_bert)
model_optimizer = torch.optim.AdamW(pre_bert.parameters(), lr=learning_rate, weight_decay=weight_decay)
#ffn_optimizer = torch.optim.Adam(ffn_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

#model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=30, gamma=0.5)
model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=50, gamma=0.5)


generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, \
        bert_batches, num_num_batches, num_label_batches, answer_label_batches, quantity_label_batches, type_label_batches, dist_batches, operator_batches = prepare_train_batch_all(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()
    result_list = []
    for idx in range(len(input_lengths)):
        pre_bert.train()
        model_optimizer.zero_grad()
        loss, result = pre_bert(bert_batches[idx], num_pos_batches[idx], num_num_batches[idx], num_label_batches[idx], answer_label_batches[idx], quantity_label_batches[idx], type_label_batches[idx], dist_batches[idx], operator_batches[idx])
        loss = loss.mean()
        result_list.append(np.array([float(ii.mean()) for ii in result]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pre_bert.parameters(), 5)
        model_optimizer.step()
        loss_total += loss.item()
        if idx % 5000 == 0:
            print("loss:", loss)
    print('train result:', sum(result_list)/len(result_list))
    model_scheduler.step()
    if epoch % 10 == 0:
        result_list = []
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, \
        bert_batches, num_num_batches, num_label_batches, answer_label_batches, quantity_label_batches, type_label_batches, dist_batches, operator_batches = prepare_train_batch_all(test_pairs, batch_size)
        with torch.no_grad():
            for idx2 in range(len(input_lengths)):
                pre_bert.eval()
                loss, result = pre_bert(bert_batches[idx2], num_pos_batches[idx2], num_num_batches[idx2], num_label_batches[idx2], answer_label_batches[idx2], quantity_label_batches[idx2], type_label_batches[idx2], dist_batches[idx2], operator_batches[idx2])
                result_list.append(np.array([float(ii.mean()) for ii in result]))
        print('test result:', sum(result_list)/len(result_list))
        pre_bert.module.bert_model.save_pretrained('./models/all_epoch_'+str(epoch))
    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")