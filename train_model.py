
"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
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
batch = 1 
model_path = '/home/alex/Desktop/TSAV_Sub4_544k.pth.tar' # path to the model
result_path = 'trained_results'# path where the result .txt files should be stored
database_path = 'aff2_processed/'  # path where the database was created (images, audio...) see create_database.py
# should be the same path


class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)


    def __len__(self):
        return len(self.indices)

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print('cpu selected!')
    # model
    model = TwoStreamAuralVisualModel(num_channels=4)
    modes = model.modes
    # load the model
    model = model.to(device)
    saved_model = torch.load(model_path, map_location=device)
    model.load_state_dict(saved_model['state_dict'])
    # disable grad, set to eval
    for p in model.parameters():
        p.requires_grad = False
    for p in model.children():
        p.train(False)



    # load dataset (first time this takes longer)
    dataset = Aff2CompDataset(database_path)
    dataset.set_modes(modes)

    # select the frames we want to process (we choose VAL and TEST)
    testvalids = np.logical_or(dataset.train_ids, dataset.train_ids)
    print('Train set length: ' + str(sum(dataset.train_ids)))
    sampler = SubsetSequentialSampler(np.nonzero(testvalids)[0])
    loader = DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=8, pin_memory=True, drop_last=False)
    output = torch.zeros((len(dataset), 17), dtype=torch.float32)
     
    # run inference
    for data in tqdm(loader):
        
        ids = data['Index'].long()

        x = {}
        for mode in modes:
            x[mode] = data[mode].to(device)

        result = model(x)
        print("answers: ")
        output[ids, :] = result.detach().cpu()  # output is EX VA AU
        # labels[ids, :] = torch.cat([ex_label, va_label, au_label], dim=1)

        print(result[0])
        print("expected")
        print(output)
        break
   