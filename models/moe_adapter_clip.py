import torch
from PIL import Image
import glob
from tqdm import tqdm
import os
import numpy as np
import math
import PIL
from models import clip
from torch.utils import data
import torch.nn.functional as F
from torch import nn
import copy
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self,seq_length):
        super(SelfAttention,self).__init__()
        self.input_size = seq_length
		
        self.Wq=nn.Linear(seq_length,seq_length)
        self.Wk=nn.Linear(seq_length,seq_length)
        
    def forward(self,input):

        
        q = self.Wq(input)
        k = self.Wk(input)
        batch_size = input.size(0)
        

        attention_scores = torch.matmul(q, k.transpose(-1,-2))/torch.sqrt(torch.tensor(float(self.input_size)).cuda())
        attention_scores = -attention_scores[:, 0, 1:]

        attention_weight = 256 * F.softmax(attention_scores, dim=-1)
        attention_weight = attention_weight.unsqueeze(2).repeat(1,1,self.input_size)
        attention_weight = torch.cat([torch.ones(batch_size, 1, self.input_size).cuda().float(), attention_weight], dim=1)
        return attention_weight * input

class ForgeryAdapter(nn.Module):
    def __init__(self, adapter_weight=0.1):
        super(ForgeryAdapter, self).__init__()
        self.adapter_weight=adapter_weight
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.sigmoid = torch.nn.Sigmoid()
        self.dwt_transform = DWTForward(J=1, wave='haar')

        self.adapters = nn.Sequential(*[nn.Sequential(
                                        nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1)
                                    ) for _ in range(4)])
                
        self.attentions = nn.Sequential(*[SelfAttention(1024) for _ in range(4)])
        
        self.fc = nn.Sequential(
                                nn.Linear(1024, 512),
                                nn.ReLU()
                            )

        self.final_fc = nn.Sequential(
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                            )

        
        
        self.freq_convs = nn.Sequential(*[nn.Sequential(
                                        nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1),
                                    ) for _ in range(4)])
        
        self.freq_attns = nn.Sequential(*[nn.MultiheadAttention(1024, num_heads=8, batch_first=True) for _ in range(4)])
        

    def forward(self, x, stage):
        # stage: [1，2，3，4]
        feature_list = []

            
        ### patch decentralization
        att_x = self.attentions[stage-1](x.transpose(0,1)).transpose(0,1)
        
        ### conv layer
        conv_x = self.adapters[stage-1](att_x[1:,:,:].permute(1,2,0)).permute(2,0,1) # (256, batch, 1024)
        
        ### DTW branch

        B = conv_x.size(1)
        H = W = int(math.sqrt(conv_x.size(0)))
        D = conv_x.size(2)
        rgb_img = conv_x.reshape(H,W,B,D).permute(2,3,0,1)
        dwt_img_l, dwt_img_h = self.dwt_transform(rgb_img)
        dwt_img = torch.cat([dwt_img_l[:, :, None], dwt_img_h[0]], dim=2)
        
        
        dwt_img = torch.mean(dwt_img, dim=2)
        dwt_img = self.freq_convs[stage-1](dwt_img)

        # cross attention fusion
        q = rgb_img.flatten(2,3).transpose(1,2) 
        k = v = dwt_img.flatten(2,3).transpose(1,2)
        fusion_x = (self.freq_attns[stage-1](q,k,v)[0] + q).transpose(0,1)
        adapt_x = torch.cat([att_x[0,:,:].unsqueeze(0), fusion_x], dim=0)
        
        
        if stage != 4:
            x = adapt_x * self.adapter_weight + x * (1-self.adapter_weight)
        else:
            x = adapt_x
        
        return x, adapt_x

        
class Router(nn.Module):
    def __init__(self, num_adapters=2):
        super(Router, self).__init__()
        self.num_adapters = num_adapters
        self.weight_generator = nn.Sequential(
                                nn.Linear(int(512*self.num_adapters), int((512*self.num_adapters)/4)),
                                nn.ReLU(),
                                nn.Linear(int((512*self.num_adapters)/4), self.num_adapters),
                            )
        self.sigmoid = torch.nn.Sigmoid()
        self.final_fc = nn.Sequential(
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                            )
    def forward(self, adapter_features):

        if len(adapter_features[0].size()) == 1:
            for i in range(len(adapter_features)):
                adapter_features[i] = adapter_features[i].unsqueeze(0)

        adapter_weight = self.sigmoid(self.weight_generator(torch.cat(adapter_features, dim=1)))
        for i in range(len(adapter_features)):
            adapter_features[i] = adapter_features[i] * adapter_weight[:, i].unsqueeze(1)
        output = self.final_fc(sum(adapter_features))

        return output


class MoE_Adapter_CLIP(nn.Module):
    def __init__(self, num_adapters = 2):
        super(MoE_Adapter_CLIP, self).__init__()
        self.num_adapters = num_adapters

        print('---- Loading CLIP Pretrained Model ----')
        device = 'cuda'
        self.backbone_model, _ = clip.load('./checkpoints/CLIP.pt', device)
        self.backbone_model.float()
        self.backbone_model.eval()
        for p in self.backbone_model.parameters():
            p.requires_grad = False
        print('---- CLIP Model Load Done ----')


        self.forgery_adapters = nn.Sequential(*[ForgeryAdapter() for _ in range(self.num_adapters)])
        

        self.router = Router(self.num_adapters)
        


    def forward_single_adapter(self, x, idx):
        adapter = self.forgery_adapters[idx]
        
        feature_list = []
        for stage in [1,2,3,4]:
            with torch.no_grad():
                x = self.backbone_model.encode_image_stage_feature(x, stage) # (257, batch, 1024)
            x, adapt_x = adapter(x, stage=stage)
            feature_list.append(adapt_x)

        feature_list = torch.stack(feature_list)
        feature = adapter.avgpool(feature_list.mean(0)[1:,:,:].permute(1,2,0)).squeeze()
        feature = adapter.fc(feature)

        return feature, adapter.final_fc(feature)


    def forward(self, x, adapter_idx=None):

        adapter_features = []
        optimized_features = []
        anchor_features = []
    
        for i in range(len(self.forgery_adapters)):
            adapter_features.append(self.forward_single_adapter(x, idx=i)[0])
    

        output = self.router(adapter_features)
        

        return output
        

