# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/4 下午4:15
 @FileName: process.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import re
import sys

from utils import DBC2SBC
import sentencepiece as spm


def clean_space():
    with open('/search/odin/bingning/data/mt/all.txt', encoding='utf-8', errors='ignore') as f:
        with open('/search/odin/bingning/data/mt/all.clean.txt', 'w', encoding='utf-8') as wf:
            for line in f:
                ss = line.strip().split('\t')
                if len(ss) != 2:
                    continue
                ss[0] = re.sub(' *', '', ss[0])
                wf.write(ss[0] + '\t' + ss[1] + '\n')


def spm_train():
    spm.SentencePieceTrainer.Train(
        '--input=/search/odin/bingning/data/AnswerSelection/qa/result.banjiao.utf8.small '
        '--model_prefix=/search/odin/bingning/data/banjiao.unigram.35000 --vocab_size=35000 --character_coverage=1 '
        '--training_sentence_size=15000000 --num_sub_iterations=1 --input_sentence_size=15000000 '
        '--model_type=unigram --num_threads=36 --max_sentence_length=200000')


#
#
def zng(paragraph):
    sents = re.findall(u'[^!?。\!\?]+[!?。\!\?]?', paragraph, flags=re.U)
    output = [sents[0]]
    for sent in sents[1:]:
        if len(list(sent)) <= 5:
            output[-1] += sent
        else:
            output.append(sent)
    return output


def proce(line):
    with open('/search/odin/bingning/data/AnswerSelection/qa/result.banjiao.utf8', 'a', encoding='utf-8') as f:
        ss = DBC2SBC(line.strip())
        outputs = zng(ss)
        st = ' '.join(outputs)
        f.write(st + '\n')


def procLixiang():
    result = []
    from utils import remove_duplciate_lst
    import numpy as np
    with open('/search/odin/bingning/data/mt/data/allResBing.txt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            cc = line.strip().split('##')
            result.append(cc[1] + '\t' + cc[0])

    with open('/search/odin/bingning/data/mt/data/allResYoudao.txt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            cc = line.strip().split('##')
            if len(cc) < 2:
                print('wrong')
                continue
            result.append(cc[1] + '\t' + cc[0])
    print(len(result))
    result = remove_duplciate_lst(result)
    print(len(result))
    np.random.shuffle(result)
    with open('/search/odin/bingning/data/mt/data/all.txt', 'w', encoding='utf-8') as f:
        for line in result:
            f.write(line + '\n')


procLixiang()

from utils import multi_process

# spm_train()
# sp = spm.SentencePieceProcessor()
# sp.load('/search/odin/bingning/data/mt/total.uni.35000.model')
#
#
# print(sp.EncodeAsPieces('We are the world to make a better place.'))
# print(sp.EncodeAsPieces('药物流产主要的适应症是妊娠49天以内无明显药物流产禁忌症者，均可以选择药物流产。'))
