
from aff2newdataset import Aff2CompDatasetNew
from tsav import TwoStreamAuralVisualModel
import torch
import torch.nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

model = TwoStreamAuralVisualModel().cuda()
model.load_state_dict(torch.load('7model.pth'))
model.eval()
train_set = Aff2CompDatasetNew(root_dir='aff2_processed',test_set=True)
loop=DataLoader(dataset=train_set,batch_size=64,shuffle=True,num_workers=8,pin_memory=True)

total = 0
correct = 0
for data in tqdm(loop):
    with torch.no_grad():
        if(int(data['expressions'][0])==-1):
            continue
        if('clip' not in data):
            continue
        x = {}
        x['clip'] = data['clip'].cuda()
        result = model(x)
        sm = torch.nn.Softmax(dim=1)
        result = sm(result)
        labels = torch.LongTensor(data['expressions']).cuda()
        result = torch.argmax(result,dim=1)
        total += labels.size(0)
        # print(result)
        # print(labels)
        correct += (result == labels).sum().item()
final_accuracy = 100*correct/total
f = open("accuracy.txt",'a')
f.write(str(final_accuracy))
f.close
