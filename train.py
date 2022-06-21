from cv2 import exp
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
import torch.optim 
import torch.nn as nn
from aff2newdataset import Aff2CompDatasetNew
wandb.init(project="expressions_estimation")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print('cpu selected!')

# device = torch.device("cpu")

batch_size = 22 
save_model_path = '/home/alex/Desktop/TSAV_Sub4_544k.pth.tar' # path to the model
database_path = 'aff2_processed/'  # path where the database was created (images, audio...) see create_database.py
epochs = 10 
clip_value = 5

train_set = Aff2CompDatasetNew(root_dir='aff2_processed')

train_loader =DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)

model = TwoStreamAuralVisualModel(num_channels=4).to(device)
modes = model.modes
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
# note about loss
# D. Loss functions
# We use the same loss functions as they are defined in [17].
# The categorical cross entropy for categorical expression clas-
# sification. The binary cross entropy for action unit detection
# and the concordance correlation coefficient loss for valence
# and arousal estimation. We divide each loss by the amount
# of labeled samples in the current mini-batch and use the sum
# as training objective.
# au_detection_metric = nn.BCELoss()
# va_metrics = audtorch.metrics.functional.concordance_cc
expression_classification_fn = nn.CrossEntropyLoss()
wandb.watch(model)
for epoch in range(epochs):
    total = 0
    correct = 0
    model.train()
    loop =tqdm(train_loader,leave=False)
    for data in loop:
        if(int(data['expressions'][0])==-1):
            continue
        if('clip' not in data):
            continue
        x = {}
        x['clip'] = data['clip'].to(device)
        optimizer.zero_grad()
        result = model(x)
        expected = torch.LongTensor(data['expressions']).to(device)
        loss = expression_classification_fn(result,expected)
        torch.save(model.state_dict(), f'{epoch}model.pth')
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())
        # print(result[1].sum())
        # total += expected.size(0)
        # correct += result.eq(expected).sum().item()
        wandb.log({
            "sum": result[0].sum()
        })
        wandb.log({
           "epoch": epoch+1,
             "train_loss": loss.item()
        })
torch.save(model.state_dict(), 'model.pth')