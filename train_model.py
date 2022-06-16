import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tsav import TwoStreamAuralVisualModel
from aff2compdataset import Aff2CompDataset
from write_labelfile import write_labelfile
from utils import ex_from_one_hot, split_EX_VA_AU
from tqdm import tqdm
import os

model_path = 'TSAV416k.pth.tar' # path to the model
result_path = 'results'# path where the result .txt files should be stored
database_path = '/home/alex/detection/ABAW2020TNT-Modified/aff2_processed'  # path where the database was created (images, audio...) see create_database.py

class SubsetSequentialSampler(Sampler):


    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)


    def __len__(self):
        return len(self.indices)

if __name__ == '__main__':
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        device = torch.device("cuda")
        print("cuda selected")
    else:
        device = torch.device("cpu")
        print('cpu selected!')
    model = TwoStreamAuralVisualModel(num_channels=4)
    dataset = Aff2CompDataset(database_path)
    dataset.set_modes(modes)


    # select the frames we want to process (we choose VAL and TEST)
    testvalids = np.logical_or(dataset.train_ids, dataset.train_ids)
    print('Train set length: ' + str(sum(dataset.train_ids)))
    sampler = SubsetSequentialSampler(np.nonzero(testvalids)[0])
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=8, pin_memory=False, drop_last=False)

    output = torch.zeros((len(dataset), 17), dtype=torch.float32)
    # labels = torch.zeros((len(dataset), 17), dtype=torch.float32)

