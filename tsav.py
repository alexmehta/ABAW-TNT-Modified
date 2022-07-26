"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


class VideoModel(nn.Module):
    def __init__(self, num_channels=3):
        super(VideoModel, self).__init__()
        self.r2plus1d = models.video.r2plus1d_18(pretrained=True)
        self.r2plus1d.fc = nn.Sequential(nn.Dropout(0.0),
                                         nn.Linear(in_features=self.r2plus1d.fc.in_features, out_features=128))
        self.r2plus1d.requires_grad_ = True
        self.modes = ["clip"]

    def forward(self, x):
        return self.r2plus1d(x)
class TwoStreamAuralVisualModel(nn.Module):
    def __init__(self, num_channels=4, audio_pretrained=False):
        super(TwoStreamAuralVisualModel, self).__init__()
        self.video_model = VideoModel(num_channels=num_channels)
        self.fc = nn.Sequential(nn.Dropout(0.0),
                                nn.ReLU(),
                                nn.Linear(in_features=self.video_model.r2plus1d.fc._modules['1'].in_features,out_features=8+12))
        self.modes = ['clip']
        self.video_model.r2plus1d.fc = Dummy()
    def forward(self, x):
        clip = x['clip']
        video_model_features = self.video_model(clip)

        features = video_model_features
        out = self.fc(features)
        return out 
