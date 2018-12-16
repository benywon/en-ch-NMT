# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/6 下午5:02
 @FileName: interface.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from werkzeug.serving import make_server
from werkzeug.wrappers import Request, Response
from jinja2 import *
import sentencepiece as spm

from model import GeneratorSelfAttention
from torchUtils import get_model, get_tensor_data

n_embedding = 768
n_hidden = 768
n_layer = 2
batch_size = 32

sp = spm.SentencePieceProcessor()
sp.load('total.uni.35000.model')
model = GeneratorSelfAttention(sp.GetPieceSize(), n_embedding, n_hidden, n_layer)
print('build done')
model.load_state_dict(
    get_model('model/model.trans.{}.{}.th'.format(n_hidden, n_layer)))
model.cuda()
model.eval()
print('model loaded')


def trim(prediction):
    if sp.GetPieceSize() + 1 in prediction:
        end = prediction.index(sp.GetPieceSize() + 1)
    elif 0 in prediction:
        end = prediction.index(0)
    else:
        end = len(prediction)
    prediction = prediction[0:end]
    return prediction


def translate(sequence):
    sequence = sequence.strip()
    ids = sp.EncodeAsIds(sequence)
    inputs = [None, torch.LongTensor([ids[0:100]]).cuda()]
    with torch.no_grad():
        prediction = model(inputs)
    output = get_tensor_data(prediction)
    prediction = trim(output[0].tolist())
    return sp.DecodeIds(prediction)


@Request.application
def application(request):
    url_path = request.path
    if url_path == '/translate':
        source = request.args['source']
        target = translate(source)
        translation_html_data = open('webservice/translate.html', 'r',
                                     encoding='utf-8').read()
        return Response(Template(translation_html_data).render(source=source, target=target), status='200 OK',
                        content_type='text/html')
    translation_html_data = open('webservice/index.html', 'rb').read()
    return Response(translation_html_data, status='200 OK', content_type='text/html')


if __name__ == '__main__':
    server = make_server('0.0.0.0', 4000, application)
    server.serve_forever()
#     from werkzeug.serving import run_simple
#
#     run_simple('0.0.0.0', 4000, application)
