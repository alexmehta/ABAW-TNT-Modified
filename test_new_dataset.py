from aff2newdataset import Aff2CompDatasetNew

dataset = Aff2CompDatasetNew(root_dir='aff2_processed')
for d in dataset:
    if(d['clip']!=None):
        print(d)
