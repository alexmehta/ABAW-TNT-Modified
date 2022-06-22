
from aff2newdataset import Aff2CompDatasetNew
from tsav import TwoStreamAuralVisualModel
import torch
model = TwoStreamAuralVisualModel().cuda()
model = model.load_state_dict('7model.pth')

train_set = Aff2CompDatasetNew(root_dir='aff2_processed',test_set=True)
total = 0
correct = 0
for data in loop:
    if(int(data['expressions'][0])==-1):
        continue
    if('clip' not in data):
        continue
    x = {}
    x['clip'] = data['clip'].to(device)
    result = model(x)
    labels = torch.LongTensor(data['expressions']).to(device)
    total += labels.size(0)
    correct += (result == labels).sum().item()

print(100*correct//total)