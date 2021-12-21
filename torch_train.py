import torch, cv2
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
from torchvision.datasets import mnist # 获取数据集
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.device("cuda")
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

train_loader = data.DataLoader(Train_val_data("./train_path.txt"),batch_size=4,shuffle=True)
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
            torch.nn.Sigmoid()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,1,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.Sigmoid()
        )
        
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(32,16,3,1,1),
            torch.nn.BatchNorm2d(16),
            torch.nn.Sigmoid()
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

model = CNNnet()
model = model.to(device)
print(model)

loss_func = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)

loss_count = []
idx=1
for epoch in range(1000):
    for i,(x,y) in enumerate(train_loader):

        # batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
        # batch_y = Variable(y) # torch.Size([128])
        # import pdb;pdb.set_trace()
        batch_x, batch_y = x.to(device), y.to(device)
        # batch_x, batch_y = x, y
        # batch_x = torch.tensor(x, dtype=torch.float64).clone().detach()
        # batch_y = torch.tensor(y, dtype=torch.float64).clone().detach()
        # model = model.double()

        out = model.forward(batch_x) # torch.Size([128,10])

        # 获取损失
        loss = loss_func(out,batch_y)
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        opt.step() # 将参数更新值施加到net的parmeters上
        if i%100 == 0:
            loss_count.append(loss)
            print('{}:\t'.format(i), loss.item())
            torch.save(model.state_dict(), './ckpt/ckpt_{}.pt'.format(idx))
            idx = idx+1
        if i % 100 == 0:
            for a,b in test_loader:
                test_x = a.cuda()
                test_y = b.cuda()
                out = model.forward(test_x)
                # print('test_out:\t',torch.max(out,1)[1])
                # print('test_y:\t',test_y)
                break
import pdb;pdb.set_trace()
plt.figure('PyTorch_CNN_Loss')
plt.plot(loss_count,label='Loss')
plt.legend()
plt.show()

# https://blog.csdn.net/qq_33039859/article/details/80934060


# https://www.cnblogs.com/expttt/p/13047451.html
# https://blog.csdn.net/jiangyutongyangyi/article/details/103583532