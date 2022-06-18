from re import X
from torch.utils.data import Dataset
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
import torchaudio
import math
import subprocess
from utils import *
from clip_transforms import *
from video import Video
import csv
class Aff2CompDatasetNew(Dataset):
    # this code here is very inefficent (but works well). 
    def add_video(self,info,extracted_frames_list):
       for folder in extracted_frames_list:
            if(folder.startswith(info['vid_name'][0])):
                image_list = os.listdir(os.path.join(self.root_dir,"extracted",folder,"mask"))
                image_list.sort()
                for i,image in enumerate(image_list):
                    if(image.startswith(info['vid_name'][1])):
                        return self.take_mask(info, folder, image_list, i, image)
    def take_mask(self, info, folder, image_list, i, image):
        before = i
        after = len(image_list) - i - 1
        info['path'] = os.path.join(self.root_dir,"extracted",folder,image)
                    #this is where some experiments come in handy (how do we want to split the 8 frames) 

        clip= np.zeros((self.clip_len, self.input_shape[0], self.input_shape[1], 4), dtype=np.uint8)
        # print(image)
        # print(i)
        # print(image_list[0])
        # print(image_list)
        if(before>=7):
                        # take last 7 frames and current frame
            for cnt,z in enumerate(range(i-8,i+1)):
                image_path = os.path.join(self.root_dir,"extracted",folder,"mask",image)
                mask_img = Image.open(image_path)
                try:
                    clip[cnt, :, :, 3] = np.array(mask_img)
                    # print("worked fine")
                except:
                    X = 0
                    # print("there was an issue, but we just leave a blask mask :)")

        else:
            return None
        return  self.clip_transform(clip)

    def __init__(self,root_dir='',mtl_path = 'mtl_data/'):
        super(Aff2CompDatasetNew,self).__init__()
        #file lists
        self.root_dir = root_dir
        self.audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])
        self.clip_transform = ComposeWithInvert([NumpyToTensor(), Normalize(mean=[0.43216, 0.394666, 0.37645, 0.5],
                                                                            std=[0.22803, 0.22145, 0.216989, 0.225])])
        self.videos =   []
        self.videos += [each for each in os.listdir(root_dir) if each.endswith(".mp4")]
        self.metadata = []
        self.metadata += [each for each in os.listdir(root_dir) if each.endswith(".json")]
        self.extracted_frames = []
        self.extracted_frames += [each for each in os.listdir(os.path.join(root_dir , "extracted"))]

        #video info
        self.clip_len = 8
        self.input_shape = (112,112)
        self.dialation = 6
        self.label_frame = self.clip_len * self.dialation
        #audio
        self.window_size = 20e-3
        self.window_stride = 10e-3
        self.sample_rate = 44100
        num_fft = 2 ** math.ceil(math.log2(self.window_size * self.sample_rate))
        window_fn = torch.hann_window
        self.sample_len_secs = 10
        self.sample_len_frames = self.sample_len_secs * self.sample_rate
        self.audio_shift_sec = 5
        self.audio_shift_samples = self.audio_shift_sec * self.sample_rate
        #transforms 

        train_csv = os.path.join(mtl_path, "train_set.txt" )
        test_csv = os.path.join(mtl_path, "test_set.txt" )
        self.training = []
        self.training += self.create_inputs(train_csv)
        # self.testing = []
        # self.testing += self.create_inputs(test_csv)
    def create_inputs(self,csv_path):
        labels = []
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=",")
            next(csv_reader)
            for row in csv_reader:
                labels.append(row)
        outputs = []
        for row in labels:
            vid_name = row[0].split('/')
            valience = row[1]
            arousal = row[2]
            expressions = row[3:(3+8)]
            action_units = row[(3+8):]
            expected_output = {}
            expected_output['vid_name'] = vid_name
            expected_output['valience'] = valience
            expected_output['arousal'] = arousal
            expected_output['expressions'] = expressions
            expected_output['action_units'] = action_units
            outputs.append(expected_output)
        return outputs
    def __getitem__(self, index):
        d = self.training[index]
        d['clip']  = self.add_video(d,self.extracted_frames)
        
        return d
    def __len__(self):
        return len(self.training)
    # def add_audio(info):


        