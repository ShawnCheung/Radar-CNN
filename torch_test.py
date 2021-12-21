import numpy as np
import cv2
import torch
from torch.utils import data # 获取迭代数据


class Train_val_data(data.Dataset):
    def __init__(self, imgPath):
        super(Train_val_data,self).__init__()
        self.paths = [line_.rstrip() for line_ in open(imgPath)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx].split()
        X = (cv2.imread("./train-k/"+path[0])/255).astype(np.float32)
        Y = (cv2.imread("./train-c/"+path[0])/255).astype(np.float32)

        return  X.transpose(2,0,1), Y.transpose(2,0,1)
test_loader = data.DataLoader(Train_val_data("./test_path.txt"),batch_size=4,shuffle=True)

class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,1,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(32,16,3,1,1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(16,3,3,1,1),
            torch.nn.BatchNorm2d(3),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.conv6(x)
        x = self.conv7(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNNnet()
model.load_state_dict(torch.load("ckpt/ckpt_600.pt"))
name = 0
for idx, (a,b) in enumerate(test_loader):
    before = a.detach().numpy().transpose(0,2,3,1)
    
    out = model.forward(a)
    out = out.detach().numpy()
    out = out.transpose(0,2,3,1)

    origin = b.detach().numpy()
    origin = origin.transpose(0,2,3,1)
    for i in range(4):
        img = np.zeros((1536,512,3))
        img[0:512,:,:] = origin[i]*255
        img[512:1024,:,:] = before[i]*255
        img[1024:1536,:,:] = out[i]*255
        name = name+1
        cv2.imwrite("./out/c-k-rc-{}.jpg".format(name), img)
    


