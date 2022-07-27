import random
import torch
from torch.utils.data.dataloader import DataLoader
from tsav import TwoStreamAuralVisualModel
from tqdm import tqdm
import wandb
import torch.optim 
import torch.nn as nn
from aff2newdataset import Aff2CompDatasetNew
from torchvision.io import write_video
wandb.init(project="full_mtl")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print('cpu selected!')
# device = torch.device("cpu")
#hyperparams
batch_size =16
save_model_path = '/home/alex/Desktop/TSAV_Sub4_544k.pth.tar' # path to the model
database_path = 'aff2_processed/'  # path where the database was created (images, audio...) see create_database.py
epochs = 30
clip_value = 5
#clean data function
def clean_dataset(set):
    z = 0
    removed_for_beginning = 0
    for i in range(set.__len__()):
        data = set.__getitem__(i) 
        if(data == None or data['clip']==None):
            i = i-1
            z+=1
        elif ( torch.equal(data['clip'],torch.zeros((3,8,112,112)))):
            i = i-1
            removed_for_beginning = removed_for_beginning+1
            z+=1
        wandb.log({"errored data": z, "i":i,"good data": i-z,"removed_for_beginning":removed_for_beginning, "removed for other reason":z-removed_for_beginning})
    return set
num_workers = 8
dataset = Aff2CompDatasetNew(root_dir='aff2_processed')
dataset = clean_dataset(dataset)
train_set, val_set = torch.utils.data.random_split(dataset,[int(dataset.__len__()*0.95),dataset.__len__() - int((dataset.__len__()*0.95))])
train_set = dataset
val_set = dataset
train_loader =DataLoader(dataset=train_set,num_workers=num_workers,batch_size=batch_size,shuffle=True)
val_loader =DataLoader(dataset=val_set,num_workers=num_workers,batch_size=8,shuffle=True)
model = TwoStreamAuralVisualModel(num_channels=3).to(device)
modes = model.modes
learning_rate = 0.0001
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
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
  "batch_size":batch_size ,
  "num_workers": num_workers
}
expression_classification_fn = nn.CrossEntropyLoss()

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
        valience = torch.DoubleTensor(data['valience']).to(device) 
        arousal = torch.DoubleTensor(data['arousal']).to(device)        
        if(torch.equal(data['clip'],torch.zeros((3,8,112,112)))):
            print("got through")
        losses = [loss_exp_0,loss_exp_1,loss_exp_2,loss_exp_3,loss_exp_4,loss_exp_5,loss_exp_6,loss_exp_7,loss_exp_8,loss_exp_9,loss_exp_10,loss_exp_11,loss_exp,valience,arousal]
        loss = losses[0]
        for l in losses[1:]:
            loss = loss + l
        wandb.log({
           "Before backprop: Total Train Loss": loss.sum().item(),
           "Before backprop: Expression Loss" : loss_exp.item(),
           "Before backprop: valience_loss": valience.sum().item(),
           "Before backprop: arousal_loss": arousal.sum().item(),
           "Before backprop: au_0" : loss_exp_0.sum().item(),
           "Before backprop: au_1": loss_exp_1.sum().item(),
           "Before backprop: au_2": loss_exp_2.sum().item(),
           "Before backprop: au_3" : loss_exp_3.sum().item(),
           "Before backprop: au_4": loss_exp_4.sum().item(),
           "Before backprop: au_5": loss_exp_5.sum().item(),
           "Before backprop: au_6": loss_exp_6.sum().item(),
           "Before backprop: au_7": loss_exp_7.sum().item(),
           "Before backprop: au_8": loss_exp_7.sum().item(),
           "Before backprop: au_9": loss_exp_7.sum().item(),
           "Before backprop: au_10": loss_exp_7.sum().item(),
           "Before backprop: au_11": loss_exp_7.sum().item(),
        #    "image": wandb.Image(x['clip'][0])
        })
        loss.sum().backward()
        optimizer.step()
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.sum().item())
        torch.save(model.state_dict(), f'{epoch+1}_model.pth')
        # (C,T,H,W)
        write_video("examples/video_A.mp4",random.choice(x['clip']).cpu().permute(1,2,3,0)[:,:,:,:].mul(255),1)
        wandb.log({
           "epoch": epoch+1,
           "Total Train Loss": loss.sum().item(),
           "Expression Loss" : loss_exp.item(),
           "valience_loss": valience.sum().item(),
           "arousal_loss": arousal.sum().item(),
           "au_0" : loss_exp_0.sum().item(),
           "au_1": loss_exp_1.sum().item(),
           "au_2": loss_exp_2.sum().item(),
           "au_3" : loss_exp_3.sum().item(),
           "au_4": loss_exp_4.sum().item(),
           "au_5": loss_exp_5.sum().item(),
           "au_6": loss_exp_6.sum().item(),
           "au_7": loss_exp_7.sum().item(),
           "au_8": loss_exp_7.sum().item(),
           "au_9": loss_exp_7.sum().item(),
           "au_10": loss_exp_7.sum().item(),
           "au_11": loss_exp_7.sum().item(),
        })
        wandb.log(
  {"video": wandb.Video("examples/video_A.mp4", fps=4, format="mp4")})
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
        valience = torch.DoubleTensor(data['valience']).to(device) 
        arousal = torch.DoubleTensor(data['arousal']).to(device)        

        losses = [loss_exp_0,loss_exp_1,loss_exp_2,loss_exp_3,loss_exp_4,loss_exp_5,loss_exp_6,loss_exp_7,loss_exp_8,loss_exp_9,loss_exp_10,loss_exp_11,loss_exp,valience,arousal]
        loss = losses[0]
        for l in losses[1:]:
            loss = loss + l
        loop.set_description(f"Epoch [{epoch+1}/{epochs}] validation")
        loop.set_postfix(loss=loss.sum().item(),    )
        wandb.log({
           "epoch_val": epoch+1,
           "val_loss_sum": loss.sum().item(),
        })
model.train()
wandb.watch(model)
for epoch in range(epochs):
    # val()
    train() 
model.eval()
torch.save(model.state_dict(), 'final_model.pth')
