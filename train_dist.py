# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/8 下午2:12
 @FileName: train_dist.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import argparse
import multiprocessing as mp
import time

import sentencepiece as spm
import torch
import torch.distributed as dist
from nltk.translate.bleu_score import sentence_bleu

from model import GeneratorSelfAttention
from torchUtils import get_model_parameters
from utils import *

torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
print(args.local_rank, dist.get_rank(), dist.get_world_size())
torch.cuda.set_device(args.local_rank)
sp = spm.SentencePieceProcessor()
sp.load('/search/odin/bingning/data/mt/total.uni.35000.model')

data_path = '/search/odin/bingning/data/mt/en.ch.30m.txt'
queue = mp.Queue(50)
num_thread = 6

n_embedding = 256
n_hidden = 256
n_layer = 4
batch_size = 64
log_interval = 500


def get_line_id(cc):
    chinese = cc[0]
    english = cc[1]
    return [sp.EncodeAsIds(chinese)[0:60], sp.EncodeAsIds(english)[0:60]]


def get_shuffle_data(data):
    pool = {}
    for one in data:
        length = len(one[1]) // 4
        if length not in pool:
            pool[length] = []
        pool[length].append(one)
    for one in pool:
        np.random.shuffle(pool[one])
    length_lst = list(pool.keys())
    random.shuffle(length_lst)
    return [x for y in length_lst for x in pool[y]]


def generate_data(thread_id):
    np.random.seed(thread_id * 10 + 20)
    base = 150
    seq_len = max(base - 10, int(np.random.normal(base, 10)))
    seq_len = min(seq_len, base + 10)

    en_or_ch = thread_id % 2 == 0
    while True:
        data = []
        number = 0
        en_or_ch = not en_or_ch
        if en_or_ch:
            first = 0
            second = 1
        else:
            first = 1
            second = 0
        with open(data_path, encoding='utf-8', errors='ignore') as f:
            for line_ in f:
                number += 1
                if number % (num_thread * dist.get_world_size()) != (
                        dist.get_rank() * num_thread + thread_id) or number < 1024 * dist.get_world_size():
                    continue
                s = line_.strip().split('\t')
                if len(s) != 2:
                    continue
                cc = get_line_id(s)
                if len(cc[0]) < 2 or len(cc[1]) < 2:
                    continue
                data.append(cc)
                if len(data) >= batch_size * seq_len:
                    data = get_shuffle_data(data)
                    np.random.seed(thread_id * seq_len + 20)
                    base = 150
                    seq_len = max(base - 10, int(np.random.normal(base, 10)))
                    seq_len = min(seq_len, base + 10)

                    for i in range(0, len(data), batch_size):
                        english, _ = padding(
                            [[sp.GetPieceSize()] + x[first] + [sp.GetPieceSize() + 1] for x in
                             data[i:i + batch_size]], max_len=60)
                        chinese, _ = padding([x[second] for x in data[i:i + batch_size]], max_len=60)
                        queue.put([torch.LongTensor(english), torch.LongTensor(chinese)])
                    data = []

            print('thread {} is done'.format(thread_id))


for one in range(num_thread):
    p = mp.Process(target=generate_data, args=(one,))
    p.start()

model = GeneratorSelfAttention(sp.GetPieceSize(), n_embedding, n_hidden, n_layer)
print('model size {}'.format(get_model_parameters(model)))
reload = True
if reload:
    with open('/search/odin/bingning/program/dis_torch/model.trans.{}.{}.th'.format(n_hidden, n_layer), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage.cuda(dist.get_rank())))
model = model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1.0e-5, lr=3.0e-4)
test_data = []
num = 0
with open(data_path, encoding='utf-8') as f:
    for line in f:
        num += 1
        if num % dist.get_world_size() == dist.get_rank():
            s = line.strip().split('\t')
            cc = get_line_id(s)
            test_data.append(cc)
            if len(test_data) >= 1024:
                break
test_data = sorted(test_data, key=lambda x: len(x[1]))


def get_one_blue(reference, prediction):
    if sp.GetPieceSize() + 1 in prediction:
        end = prediction.index(sp.GetPieceSize() + 1)
    elif 0 in prediction:
        end = prediction.index(0)
    else:
        end = len(prediction)
    prediction = prediction[0:end]

    return sentence_bleu([reference], prediction)


def metric_average(val):
    tensor = torch.tensor(val).cuda()
    dist.reduce(tensor, 0)
    return tensor.item()


def test():
    model.eval()
    result = []
    with torch.no_grad():
        for j in range(0, len(test_data), batch_size):
            source, _ = padding([x[0] for x in test_data[j:j + batch_size]], max_len=60)
            target = [x[1] for x in test_data[j:j + batch_size]]
            output = model([None, torch.LongTensor(source).cuda()])
            for pre, tru in zip(output.cpu().data.numpy().tolist(), target):
                result.append(get_one_blue(tru, pre))
    bleu = np.mean(result)
    return metric_average(bleu)


def train(sent_processed):
    model.train()
    num_ = 0
    total_loss_mask = 0
    pre_time = None
    current_rank_processed = 0
    while True:
        one = queue.get()
        optimizer.zero_grad()
        loss = model([x.cuda() for x in one])
        # if np.isnan(loss.item()) or np.isinf(loss.item()):
        #     continue
        loss.backward()
        # optimizer.backward(loss)
        optimizer.step()
        current_rank_processed += one[0].size(0)
        total_loss_mask += loss.item() * one[0].size(0)
        # print('{} is {}'.format(dist.get_rank(), loss.item()))
        num_ += 1
        if num_ % log_interval == 0:
            torch.cuda.empty_cache()
            if pre_time is None:
                eclipse = 0
            else:
                eclipse = time.time() - pre_time
            total_loss_mask = metric_average(total_loss_mask)
            current_rank_processed = metric_average(current_rank_processed * 1.0)
            sent_processed += current_rank_processed
            if dist.get_rank() == 0:
                print(
                    'mask loss is {:5.4f},ms per sentence is {:7.4f}, sent processed {:g}'.format(
                        total_loss_mask / current_rank_processed,
                        1000 * eclipse / current_rank_processed,
                        sent_processed))
            pre_time = time.time()
            total_loss_mask = 0.0
            current_rank_processed = 0
        if num_ == log_interval * 10:
            break
    return sent_processed


best_acc = 0
n = 0
for epoch in range(10000):
    n = train(n)
    ppl = test()
    if epoch > 5:
        for g in optimizer.param_groups:
            g['lr'] *= 0.95
    if dist.get_rank() == 0:
        ppl /= dist.get_world_size()
        if ppl > best_acc:
            best_acc = ppl
        with open('model.trans.{}.{}.th'.format(n_hidden, n_layer),
                  'wb') as f:
            torch.save(model.module.state_dict(), f)
        print('----------------epoch {} current ppl {:6.4f}, best ppl {:6.4f}--------------'.format(epoch, ppl * 100,
                                                                                                    best_acc * 100))
