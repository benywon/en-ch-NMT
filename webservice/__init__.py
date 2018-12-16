# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/6 下午5:09
 @FileName: __init__.py.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

from torch.nn import functional as F
import torch

print(F.softmax(torch.tensor([1.0,2.0,3.0])))
print(F.softmax(torch.tensor([2.0,4.0,6.0])))