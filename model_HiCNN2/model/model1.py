
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.net1_conv1 = nn.Conv2d(1, 64, 13)
    self.net1_conv2 = nn.Conv2d(64, 64, 1)
    self.net1_conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    self.net1_conv4R = nn.Conv2d(128, 128, 3, padding=1, bias=False)
    self.net1_conv5 = nn.Conv2d(128*25, 1000, 1, padding=0, bias=True)
    self.net1_conv6 = nn.Conv2d(1000,64,1,padding=0, bias=True)
    self.net1_conv7 = nn.Conv2d(64,1,3,padding=1, bias=False)
    
    self.relu = nn.ReLU(inplace=True)

    self.weights = nn.Parameter((torch.ones(1, 2)/2), requires_grad=True)
    # He initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def forward(self, input):
    # ConvNet1
    x = self.relu(self.net1_conv1(input))
    x = self.relu(self.net1_conv2(x))
    residual = x
    x2 = self.net1_conv3(x)
    output1 = x2
    outtmp = []
    for i in range(25):
      output1 = self.net1_conv4R(self.relu(self.net1_conv4R(self.relu(output1))))
      output1 = torch.add(output1, x2)
      outtmp.append(output1)
    output1 = torch.cat(outtmp, 1)
    output1 = self.net1_conv5(output1) 
    output1 = self.net1_conv6(output1)
    output1 = torch.add(output1, residual)
    output1 = self.net1_conv7(output1)

    return output1

