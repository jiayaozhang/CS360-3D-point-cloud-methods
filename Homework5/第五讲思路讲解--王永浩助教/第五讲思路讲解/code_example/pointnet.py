import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_utils as pt_utils
from typing import List


class PointNet(nn.Module):
    def __init__(self,*, mlp1:List[int], mlp2:List[int],segmentaion: bool = False, tailmlp:List[int] = None):
        '''
        para mlp: list of int
        para bn: whether to use batchnorm
        '''
        super().__init__()
        self.mlp1 = pt_utils.SharedMLP(mlp1,bn = False)
        self.mlp2 = pt_utils.SharedMLP(mlp2,bn = False)
        self.useforseg = segmentaion
        if segmentaion:
            seg_layers = []
            for k in range(0,tailmlp.__len__()-2):
                seg_layers.append(pt_utils.Conv1d(
                    tailmlp[k],
                    tailmlp[k+1],
                    bn = True
                ))
            seg_layers.append(pt_utils.Conv1d(tailmlp[-2],tailmlp[-1],activation=None))
            self.segmlp = nn.Sequential(*seg_layers)
        else:
            cls_layers = []
            for k in range(0,tailmlp.__len__()-2):
                cls_layers.append(pt_utils.Conv1d(
                    tailmlp[k],
                    tailmlp[k+1],
                    bn = True
                ))
            cls_layers.append(pt_utils.Conv1d(tailmlp[-2],tailmlp[-1],activation=None))
            self.clsmlp = nn.Sequential(*cls_layers)

    def forward(self,x):
        n_pts = x.size()[1]
        if self.useforseg:
            x = self.mlp1(x)
            gl_feature = self.mlp2(x)
            gl_feature = torch.max(gl_feature,2,keepdim=True)[0]
            gl_feature = gl_feature.view(-1,1024,1).repeat(1,1,n_pts)
            gl_feature = torch.cat([x,gl_feature],1)
            gl_feature = self.segmlp(gl_feature)
        else:
            x = self.mlp1(x)
            x = self.mlp2(x)
            gl_feature = torch.max(x,2,keepdim=True)[0]
            gl_feature = self.clsmlp(gl_feature)

        return gl_feature


if __name__ == "__main__":
    k = 40
    MLP = [[3,64,64],[64,64,128,1024]]

    CLSMLP = [1024,512,256,k]

    def getmodel(segmentation: bool = False):
        return PointNet(mlp1 = MLP[0],mlp2 = MLP[1],tailmlp=CLSMLP)

    x = torch.rand(8,3,1000)
    model = getmodel()
    y = model(x)
    y = y.view(-1)
    print(y.shape)
    

    