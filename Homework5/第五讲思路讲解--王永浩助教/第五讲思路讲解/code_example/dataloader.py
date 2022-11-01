from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ModelNet40DataSet(Dataset):

    def __init__(self,traindata = True):
        self.train = traindata
        if traindata:
            self.data = np.load("train_data.npy").transpose(0,2,1)
            self.label = np.load('train_label_data.npy')
        else:
            self.data = np.load("test_data.npy").transpose(0,2,1)
            self.label = np.load('test_label_data.npy')
        

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        if self.train:
            sample_data = self.transform(sample_data)
        
        sample = {'pts_input':sample_data,'cls_labels':np.array(self.label[idx])}

        return sample
    
    def transform(self,data):
        ROT_RANGE = 1
        angle = np.random.uniform(-np.pi / ROT_RANGE, np.pi / ROT_RANGE)
        data = self._rotateAlongZ(data,angle)
        return data

    def _rotateAlongZ(self,data,rot_angle):
        cosval = np.cos(rot_angle)
        sinval = np.sin(rot_angle)
        rot_mat = np.array([[cosval,-sinval],[sinval,cosval]])
        data[[0,1],:] = np.dot(np.transpose(rot_mat),data[[0,1],:])
        return data
