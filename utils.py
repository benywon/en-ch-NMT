# -*- coding: utf-8 -*-
"""
 @Time    : 2018/7/17 下午2:42
 @FileName: utils.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import itertools
import multiprocessing
import pickle
import random

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
np.random.seed(10245)


def DBC2SBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0021 <= inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring


def SBC2DBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x0020:
            inside_code = 0x3000
        else:
            if not (0x0021 <= inside_code <= 0x7e):
                rstring += uchar
                continue
        inside_code += 0xfee0
        rstring += chr(inside_code)
    return rstring


def id_lst_to_string(id_lst, id2word):
    return ''.join([id2word[x] for x in id_lst])


def write_lst_to_file(lst, filename):
    output = '\n'.join(lst)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output)


def dump_file(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)


def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_shuffle_data(data, dim=0):
    pool = {}
    for one in data:
        length = len(one[dim])
        if length not in pool:
            pool[length] = []
        pool[length].append(one)
    for one in pool:
        np.random.shuffle(pool[one])
    length_lst = list(pool.keys())
    random.shuffle(length_lst)
    return [x for y in length_lst for x in pool[y]]


def remove_duplciate_lst(lst):
    lst.sort()
    return list(k for k, _ in itertools.groupby(lst))


def padding(sequence, pads=0, max_len=None, dtype='int32', return_matrix_for_size=False):
    # we should judge the rank
    if True or isinstance(sequence[0], list):
        v_length = [len(x) for x in sequence]  # every sequence length
        seq_max_len = max(v_length)
        if (max_len is None) or (max_len > seq_max_len):
            max_len = seq_max_len
        v_length = list(map(lambda z: z if z <= max_len else max_len, v_length))
        x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
        for idx, s in enumerate(sequence):
            trunc = s[:max_len]
            x[idx, :len(trunc)] = trunc
        if return_matrix_for_size:
            v_matrix = np.asanyarray([map(lambda item: 1 if item < line else 0, range(max_len)) for line in v_length],
                                     dtype=dtype)
            return x, v_matrix
        return x, np.asarray(v_length, dtype='int32')
    else:
        seq_len = len(sequence)
        if max_len is None:
            max_len = seq_len
        v_vector = sequence + [0] * (max_len - seq_len)
        padded_vector = np.asarray(v_vector, dtype=dtype)
        v_index = [1] * seq_len + [0] * (max_len - seq_len)
        padded_index = np.asanyarray(v_index, dtype=dtype)
        return padded_vector, padded_index

def add2count(value, map):
    if value not in map:
        map[value] = 0
    map[value] += 1


import os


def get_dir_files(dirname):
    L = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            L.append(os.path.join(root, file))
    return L


def multi_process(func, lst, num_cores=multiprocessing.cpu_count()):
    workers = Parallel(n_jobs=num_cores, backend='multiprocessing')
    output = workers(delayed(func)(one) for one in lst)
    return output


def count_file(filename):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(filename, "r", encoding="utf-8", errors='ignore') as f:
        num = (sum(bl.count("\n") for bl in blocks(f)))
    return num


def lst2str(lst):
    return ' '.join(list(map(str, lst)))


def str2lst(string):
    return list(map(int, string.split()))


def reverse_map(maps):
    return {v: k for k, v in maps.items()}
