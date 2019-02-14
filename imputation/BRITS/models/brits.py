import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

from preprocess.BRITS.models.rits import RitsModel
from sklearn import metrics
import numpy as np


class BRITSModel(nn.Module):
    def __init__(self, rnn_hid_size=10, impute_weight=10, label_weight=10,
                 num_of_units=20, drop_out=0.25, seq_len=48, include_lable=False):
        
        super(BRITSModel, self).__init__()
        
        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.num_of_units = num_of_units
        self.include_lable = include_lable
        
        self.drop_out = drop_out
        self.seq_len = seq_len
        
        self.build()
    
    def build(self):
        self.rits_f = RitsModel(self.rnn_hid_size, self.impute_weight, self.label_weight,
                                num_of_units=self.num_of_units, drop_out=self.drop_out,
                                seq_len=self.seq_len, include_lable=self.include_lable)
        
        self.rits_b = RitsModel(self.rnn_hid_size, self.impute_weight, self.label_weight,
                                num_of_units=self.num_of_units, drop_out=self.drop_out,
                                seq_len=self.seq_len, include_lable=self.include_lable)
    
    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))
        
        ret = self.merge_ret(ret_f, ret_b)
        ret['original'] = data['original']
        ret['masks'] = data['forward']['masks']
        
        return ret
    
    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])
        
        loss = loss_f + loss_b + loss_c
        
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2
        
        ret_f['loss'] = loss
        ret_f['imputations'] = imputations
        
        if self.include_lable is True:
            predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
            ret_f['predictions'] = predictions
        
        return ret_f
    
    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)
            
            if torch.cuda.is_available():
                indices = indices.cuda()
            
            return tensor_.index_select(1, indices)
        
        for key in ret:
            ret[key] = reverse_tensor(ret[key])
        
        return ret
    
    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss
    
    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)
        
        if np.isnan(np.array(ret['loss'].data)):
            return None

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
