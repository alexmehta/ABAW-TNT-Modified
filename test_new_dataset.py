from aff2newdataset import Aff2CompDatasetNew

dataset = Aff2CompDatasetNew(root_dir='aff2_processed')
print(dataset.__getitem__(0))
# for d in dataset:
    # if(d['clip']!=None):
        # print(d)
