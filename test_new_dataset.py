
from aff2newdataset import Aff2CompDatasetNew
import torchvision

train_set = Aff2CompDatasetNew(root_dir='aff2_processed')

video_tensor = train_set.__getitem__(10)['clip']
# print(video_tensor.size())
video_tensor = video_tensor.permute(1,2,3,0)
# print(video_tensor.size())
# print(video_tensor)
torchvision.io.write_video(filename="test_video.mp4",video_array=video_tensor[:,:,:,0:3] ,fps=3)
