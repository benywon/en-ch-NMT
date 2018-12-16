# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/5 下午3:44
 @FileName: inference.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import sentencepiece as spm
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from utils import write_lst_to_file
from model import Generator, GeneratorSingle
from torchUtils import *
from utils import padding

n_embedding = 512
n_hidden = 512
n_layer = 6
batch_size = 128

sp = spm.SentencePieceProcessor()
sp.load('/search/odin/bingning/data/mt/total.uni.35000.model')

data_path = '/search/odin/bingning/data/mt/all.clean.txt'

test_data = []


def get_line_id(cc):
    chinese = cc[0]
    english = cc[1]
    return [sp.EncodeAsIds(chinese)[0:60], sp.EncodeAsIds(english)]


num = 0
with open(data_path, encoding='utf-8') as f:
    for line in f:
        num += 1
        s = line.strip().split('\t')
        cc = get_line_id(s)
        test_data.append(cc)
        if len(test_data) >= 1024 * 4:
            break

test_data = sorted(test_data, key=lambda x: len(x[1]))

print('test data size is {}'.format(len(test_data)))

model = GeneratorSingle(sp.GetPieceSize(), n_embedding, n_hidden, n_layer)
model.load_state_dict(get_model('model.trans.{}.{}.th'.format(n_hidden, n_layer)))
model.cuda()
model.eval()

result = []


def get_one_blue(reference, prediction):
    if sp.GetPieceSize() + 1 in prediction:
        end = prediction.index(sp.GetPieceSize() + 1)
    elif 0 in prediction:
        end = prediction.index(0)
    else:
        end = len(prediction)
    prediction = prediction[0:end]

    return sentence_bleu([reference], prediction), prediction


translations = []
with torch.no_grad():
    for j in tqdm(range(0, len(test_data), batch_size)):
        source_ = [x[1] for x in test_data[j:j + batch_size]]
        source, _ = padding(source_, max_len=60)
        target = [x[0] for x in test_data[j:j + batch_size]]
        output = model([None, torch.LongTensor(source).cuda()])
        for pre, tru, src in zip(output.cpu().data.numpy().tolist(), target, source_):
            blue_score, processed_pre = get_one_blue(tru, pre)
            result.append(blue_score)
            translations.extend([sp.DecodeIds(src), sp.DecodeIds(tru), sp.DecodeIds(processed_pre)])
            translations.append('****' * 30)

print(np.mean(result))
write_lst_to_file(translations, 'translation.{}.{}.txt'.format(n_hidden,n_layer))
