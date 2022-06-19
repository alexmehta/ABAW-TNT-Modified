import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tsav import TwoStreamAuralVisualModel
from aff2compdataset import Aff2CompDataset
from write_labelfile import write_labelfile
from utils import ex_from_one_hot, split_EX_VA_AU
from tqdm import tqdm
import wandb
import os
import torch.optim 
import torch.nn as nn
from aff2newdataset import Aff2CompDatasetNew

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print('cpu selected!')

batch_size = 1 
save_model_path = '/home/alex/Desktop/TSAV_Sub4_544k.pth.tar' # path to the model
database_path = 'aff2_processed/'  # path where the database was created (images, audio...) see create_database.py
epochs =1 

train_set = Aff2CompDatasetNew(root_dir='aff2_processed')
train_loader = DataLoader(dataset=train_set,batch_size=batch_size)

model = TwoStreamAuralVisualModel(num_channels=4).cuda()
modes = model.modes
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
print(train_set.__getitem__(0)['audio'].size())
for epoch in range(epochs):
    for data in tqdm(train_loader):
        print(data['audio'].size())
        x = {}
        for mode in modes:
            x[mode] = data[mode].to(device)
        optimizer.zero_grad()
        result = model(x)
        expected= data['valience'] + data['arousal']+data['expressions']+data['action_units'] 
        for i,v in enumerate(expected):
            if(type(v) is tuple):
                expected[i] = v[0]
                print(type(expected[i]))
            expected[i] = float(expected[i])
        expected = torch.FloatTensor(expected).cuda()
        expected = expected.unsqueeze(0)
        print(expected.size())
        print(result.size())
        loss = loss_fn(result,expected ) 
        loss.backward()
        optimizer.step()

        break
