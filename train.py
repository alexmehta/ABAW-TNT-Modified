from cmath import nan
from cv2 import exp
from matplotlib.pyplot import axis
import torch
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F 
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tsav import TwoStreamAuralVisualModel
from aff2compdataset import Aff2CompDataset
from write_labelfile import write_labelfile
from utils import ex_from_one_hot, split_EX_VA_AU
from tqdm import tqdm
import wandb
import torch.optim 
import torch.nn as nn
from aff2newdataset import Aff2CompDatasetNew

wandb.init(project="full_mtl")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print('cpu selected!')

# device = torch.device("cpu")

batch_size =16
save_model_path = '/home/alex/Desktop/TSAV_Sub4_544k.pth.tar' # path to the model
database_path = 'aff2_processed/'  # path where the database was created (images, audio...) see create_database.py
epochs = 30
clip_value = 5

train_set = Aff2CompDatasetNew(root_dir='aff2_processed')
# print(len(train_set))
# print([train_set.__len__()*0.8,(train_set.__len__()*0.2)])
train_set, val_set = torch.utils.data.random_split(train_set,[int(train_set.__len__()*0.95),train_set.__len__() - int((train_set.__len__()*0.95))])

train_loader =DataLoader(dataset=train_set,num_workers=8,batch_size=batch_size,shuffle=True)

val_loader =DataLoader(dataset=val_set,num_workers=8,batch_size=8,shuffle=True)
model = TwoStreamAuralVisualModel(num_channels=4).to(device)
modes = model.modes
learning_rate = 0.0001
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
# note about loss
# D. Loss functions
# We use the same loss functions as they are defined in [17].
# The categorical cross entropy for categorical expression clas-
# sification. The binary cross entropy for action unit detection
# and the concordance correlation coefficient loss for valence
# and arousal estimation. We divide each loss by the amount
# of labeled samples in the current mini-batch and use the sum
# as training objective.
au_detection_metric = nn.BCEWithLogitsLoss()
def CCCLoss(x, y):
    z = torch.cat((x,y))
    p = (torch.add(x, y) + (x.mean() - y.mean())**2)
    a = torch.div(z,p)
    ccc = 2*torch.cov(a)
    return ccc    
wandb.config = {
  "learning_rate": learning_rate,
  "epochs": epochs,
  "batch_size":batch_size 
}
# def ccc_loss(output,target):
#     return 
# va_metrics = audtorch.metrics.functional.concordance_cc
expression_classification_fn = nn.CrossEntropyLoss()
wandb.watch(model)

from torchviz import make_dot

def train():
    loop =tqdm(train_loader,leave=False)
    for data in loop:
        optimizer.zero_grad()
        if(int(data['expressions'][0])==-1):
            continue
        if('clip' not in data):
            continue
        x = {}
        x['clip'] = data['clip'].to(device)
        result = model(x).to(device)
        expected = torch.LongTensor(data['expressions']).to(device)    
        loss_exp = expression_classification_fn(result[:,0:8],expected)
        au0 = torch.LongTensor(data['au0']).to(device)
        au1 = torch.LongTensor(data['au1']).to(device)
        au2 = torch.LongTensor(data['au2']).to(device)
        au3 = torch.LongTensor(data['au3']).to(device)
        au4 = torch.LongTensor(data['au4']).to(device)
        au5 = torch.LongTensor(data['au5']).to(device)
        au6 = torch.LongTensor(data['au6']).to(device)
        au7 = torch.LongTensor(data['au7']).to(device)
        au8 = torch.LongTensor(data['au8']).to(device)
        au9 = torch.LongTensor(data['au9']).to(device)
        au10 = torch.LongTensor(data['au10']).to(device)
        au11 = torch.LongTensor(data['au11']).to(device)
        loss_exp_0 = au_detection_metric(result[:,8],au0.float()).to(device)
        loss_exp_1 = au_detection_metric(result[:,9],au1.float()).to(device)
        loss_exp_2 =au_detection_metric(result[:,10],au2.float()).to(device)
        loss_exp_3 = au_detection_metric(result[:,11],au3.float()).to(device)
        loss_exp_4 = au_detection_metric(result[:,12],au4.float()).to(device)
        loss_exp_5= au_detection_metric(result[:,13],au5.float()).to(device)
        loss_exp_6 = au_detection_metric(result[:,14],au6.float()).to(device)
        loss_exp_7= au_detection_metric(result[:,15],au7.float()).to(device)
        loss_exp_8 = au_detection_metric(result[:,16],au8.float()).to(device)
        loss_exp_9 = au_detection_metric(result[:,17],au9.float()).to(device)
        loss_exp_10 =au_detection_metric(result[:,18],au10.float()).to(device)
        loss_exp_11 = au_detection_metric(result[:,19],au11.float()).to(device)
        loss = torch.Tensor(0).to(device)
        losses = [loss_exp_0,loss_exp_1,loss_exp_2,loss_exp_3,loss_exp_4,loss_exp_5,loss_exp_6,loss_exp_7,loss_exp_8,loss_exp_9,loss_exp_10,loss_exp_11,loss_exp,]
        loss = sum(losses)
     
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.sum().item())
        torch.save(model.state_dict(), f'{epoch+1}_model.pth')
        wandb.log({
           "epoch": epoch+1,
           "train_loss": loss.sum().item(),
        })
def val():
    loop =tqdm(val_loader,leave=False)
    i = 0
    for data in loop:

        if(int(data['expressions'][0])==-1):
            continue
        if('clip' not in data):
            continue
        x = {}
        x['clip'] = data['clip'].to(device)
        result = model(x).to(device)
        i+=1
        expected = torch.LongTensor(data['expressions']).to(device)    
        loss_exp = expression_classification_fn(result[:,0:8],expected)
        au0 = torch.LongTensor(data['au0']).to(device)
        au1 = torch.LongTensor(data['au1']).to(device)
        au2 = torch.LongTensor(data['au2']).to(device)
        au3 = torch.LongTensor(data['au3']).to(device)
        au4 = torch.LongTensor(data['au4']).to(device)
        au5 = torch.LongTensor(data['au5']).to(device)
        au6 = torch.LongTensor(data['au6']).to(device)
        au7 = torch.LongTensor(data['au7']).to(device)
        au8 = torch.LongTensor(data['au8']).to(device)
        au9 = torch.LongTensor(data['au9']).to(device)
        au10 = torch.LongTensor(data['au10']).to(device)
        au11 = torch.LongTensor(data['au11']).to(device)
        loss_exp_0 = au_detection_metric(result[:,8],au0.float()).to(device)
        loss_exp_1 = au_detection_metric(result[:,9],au1.float()).to(device)
        loss_exp_2 =au_detection_metric(result[:,10],au2.float()).to(device)
        loss_exp_3 = au_detection_metric(result[:,11],au3.float()).to(device)
        loss_exp_4 = au_detection_metric(result[:,12],au4.float()).to(device)
        loss_exp_5= au_detection_metric(result[:,13],au5.float()).to(device)
        loss_exp_6 = au_detection_metric(result[:,14],au6.float()).to(device)
        loss_exp_7= au_detection_metric(result[:,15],au7.float()).to(device)
        loss_exp_8 = au_detection_metric(result[:,16],au8.float()).to(device)
        loss_exp_9 = au_detection_metric(result[:,17],au9.float()).to(device)
        loss_exp_10 =au_detection_metric(result[:,18],au10.float()).to(device)
        loss_exp_11 = au_detection_metric(result[:,19],au11.float()).to(device)


        loss = torch.Tensor(0).to(device)
        losses = [loss_exp_0,loss_exp_1,loss_exp_2,loss_exp_3,loss_exp_4,loss_exp_5,loss_exp_6,loss_exp_7,loss_exp_8,loss_exp_9,loss_exp_10,loss_exp_11,loss_exp]

        loss = sum(losses)
        if torch.isnan(loss):
            print(losses[len(losses)-2])
        loop.set_description(f"Epoch [{epoch+1}/{epochs}] validation")
        loop.set_postfix(loss=loss.sum().item(),    )
        wandb.log({
           "epoch_val": epoch+1,
           "val_loss": loss.sum().item(),
        })
model.train()
for epoch in range(epochs):
    val()
    train() 
        
torch.save(model.state_dict(), 'model.pth')
