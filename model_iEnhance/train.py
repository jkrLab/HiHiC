import torch as t
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.SSIM import ssim

import numpy as np
from math import log10
from normga4 import Construct

from torchvision.models.vgg import vgg16

import sys
from tqdm import tqdm

################################################## Added by HiHiC ######
import os, time, datetime, argparse, random ############################

seed = 13  
random.seed(seed)  # Python 기본 랜덤 시드
np.random.seed(seed)  # NumPy 랜덤 시드
t.manual_seed(seed)  # CPU 랜덤 시드

if t.cuda.is_available():
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)  # 모든 GPU에 시드 적용

parser = argparse.ArgumentParser(description='iEnhance training process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('--train_option', type=str, required=True)
required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True,
                      help='HiHiC directory')
required.add_argument('--model', type=str, metavar='iEnhance', required=True,
                      help='model name')
required.add_argument('--epoch', type=int, default=128, metavar='[2]', required=True,
                      help='training epoch (default: 128)')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True,
                      help='input batch size for training (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, 
                      help='GPU ID for training (defalut: 0)')
required.add_argument('--output_model_dir', type=str, default='./checkpoints_iEnhance', metavar='[5]', required=True,
                      help='directory path of training model (default: HiHiC/checkpoints_iEnhance/)')
required.add_argument('--loss_log_dir', type=str, default='./log', metavar='[6]', required=True,
                      help='directory path of training log (default: HiHiC/log/)')
required.add_argument('--train_data_dir', type=str, metavar='[7]', required=True,
                      help='directory path of training data')
optional.add_argument('--valid_data_dir', type=str, metavar='[8]',
                      help="directory path of validation data, but hicplus doesn't need")
args = parser.parse_args()

start = time.time()

train_epoch = [] 
train_loss = []
valid_loss = []
train_time = []

def compute_initial_loss(model, data_loader, device):
    model.eval()  # 평가 모드로 전환하여 학습 관련 레이어 비활성화
    total_loss = 0.0
    num_batches = 0

    # 손실 함수 정의
    creMSE = nn.MSELoss()
    creMAE = nn.L1Loss()
    tvloss = TVLoss()
    loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()  # VGG 네트워크 불러오기
    for param in loss_network.parameters():
        param.requires_grad = False

    with t.no_grad():  # 그래디언트 계산 비활성화
        for inputs, targets in data_loader:
            # inputs, targets = batch
            inputs = inputs.to(device).unsqueeze(dim = 1).to(t.float32)
            targets = targets.to(device).unsqueeze(dim = 1).to(t.float32)

            # 모델 출력 계산
            outputs = model(inputs)

            # 각 손실 계산
            mseloss = creMSE(outputs, targets)
            maeloss = creMAE(outputs, targets)
            out_feat = loss_network(outputs.cpu().repeat([1, 3, 1, 1]))
            target_feat = loss_network(targets.cpu().repeat([1, 3, 1, 1]))
            perception_loss = creMSE(out_feat.reshape(out_feat.size(0), -1),
                                      target_feat.reshape(target_feat.size(0), -1))
            tv_loss_value = tvloss(outputs.cpu()).to(device)

            # 최종 손실 계산
            tloss = 0.5 * mseloss + 0.5 * maeloss + perception_loss.to(device) * 0.002 + tv_loss_value * 2e-8

            # 전체 손실 합산
            total_loss += tloss.item()
            num_batches += 1

    # 평균 손실 계산
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


os.makedirs(args.output_model_dir, exist_ok=True) ######################
########################################################################


class Config():

    ## trainfp = "gm64_train.npz"
    ## testfp = "gm64_test.npz"
    ## logn = 'log-train.txt'
    ## node = 'node4'

    # epoc_num = 500
    # batch_size = 8
    epoc_num = args.epoch
    batch_size = args.batch_size
    lr = 0.0003
    decay = 0.6
    accmu = 4

def wlog(fname,w):
    with open(fname,'a+') as f:
        f.write(w)
        f.close()

cfg = Config()
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
# cTrain = sys.argv[1]
cTrain = args.train_option
# t.autograd.set_detect_anomaly(True)

## ag = "node:%s\nlr:%f\ndevice:%s\nbatch size:%d\nepoc:%d\n" % \
##         (cfg.node,cfg.lr,device,cfg.batch_size,cfg.epoc_num)
## wlog(cfg.logn,ag)

# class MyData(Dataset):
#     def __init__(self,fp) -> None:

#         # rdata = np.load(fp)
#         self.lr = rdata['lr_sample']
#         self.hr = rdata['hr_sample']

#     def __getitem__(self, index):

#         return self.lr[index],self.hr[index]

#     def __len__(self):
        
#         return self.lr.shape[0]
    
class MyData_train(Dataset): ################################################################################ Added by HiHiC ##
    def __init__(self,fp) -> None: ############################################################################################

        # rdata = np.load(fp)
        data_all = [np.load(os.path.join(args.train_data_dir, fname), allow_pickle=True) for fname in fp] 
        rdata = {'data': [], 'target': []} 
        for data in data_all:
            for k, v in data.items():
                if k in rdata: 
                    rdata[k].append(v) 
        rdata = {k: np.concatenate(v, axis=0) for k, v in rdata.items()} 
        self.lr = rdata['data']
        self.hr = rdata['target']

    def __getitem__(self, index):

        return self.lr[index],self.hr[index]

    def __len__(self):
        
        return self.lr.shape[0]
    
class MyData_valid(Dataset):
    def __init__(self,fp) -> None:

        # rdata = np.load(fp)
        data_all = [np.load(os.path.join(args.valid_data_dir, fname), allow_pickle=True) for fname in fp] 
        rdata = {'data': [], 'target': []} 
        for data in data_all:
            for k, v in data.items():
                if k in rdata: 
                    rdata[k].append(v) 
        rdata = {k: np.concatenate(v, axis=0) for k, v in rdata.items()} 
        self.lr = rdata['data']
        self.hr = rdata['target']

    def __getitem__(self, index):

        return self.lr[index],self.hr[index]

    def __len__(self):
        
        return self.lr.shape[0] ###############################################################################################
###############################################################################################################################

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b, c, h, w = x.shape
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = t.pow((x[:, :, 1:, :] - x[:, :, :h-1, :]), 2).sum()
        w_tv = t.pow((x[:, :, :, 1:] - x[:, :, :, :w-1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

    @staticmethod
    def tensor_size(t1):
        return t1.size()[1] * t1.size()[2] * t1.size()[3]

if(cTrain == "0"):
    mdl = Construct().to(device)
    print("We train model from 0.")
elif(cTrain == '1'):
    print("We continue train model...\n")
    mdl = t.load("msave/mdl-best.pt")

creMSE = nn.MSELoss(size_average = True)
creMAE = nn.L1Loss(size_average = True)
tvloss = TVLoss()
vgg = vgg16(pretrained=True)
loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
for param in loss_network.parameters():
    param.requires_grad = False

optimizer = t.optim.Adam(mdl.parameters(),lr = cfg.lr)
lrdecay = t.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=80, gamma=cfg.decay)

# idata = MyData(cfg.trainfp)
# jdata = MyData(cfg.testfp)
idata = MyData_train(os.listdir(args.train_data_dir))
jdata = MyData_valid(os.listdir(args.valid_data_dir))

train_loader = DataLoader(idata,shuffle=True,batch_size=cfg.batch_size,drop_last = True)
val_loader = DataLoader(jdata,shuffle=True,batch_size=cfg.batch_size)

##############################################################################################################################
initial_train_loss = compute_initial_loss(mdl, train_loader, device) #########################################################
initial_valid_loss = compute_initial_loss(mdl, val_loader, device)
train_epoch.append(int(0))
train_time.append("0.00.00")
train_loss.append(f"{initial_train_loss:.10f}")        
valid_loss.append(f"{initial_valid_loss:.10f}") ##############################################################################
np.save(os.path.join(args.loss_log_dir, f'train_loss_{args.model}'), [train_epoch, train_time, train_loss, valid_loss]) ######


# epoc
best_ssim = 0
for e in range(cfg.epoc_num):
    
    mdl.train()
    train_bar = tqdm(train_loader)
    for i,(lrdata,hrdata) in enumerate(train_bar):

        lrdata = lrdata.to(device).unsqueeze(dim = 1).to(t.float32)
        hrdata = hrdata.to(device).unsqueeze(dim = 1).to(t.float32)

        # Train
        fake_hr = mdl(lrdata)
        # 3-loss
        mseloss = creMSE(fake_hr,hrdata)
        maeloss = creMAE(fake_hr,hrdata)
        out_feat = loss_network(fake_hr.cpu().repeat([1,3,1,1]))
        target_feat = loss_network(hrdata.cpu().repeat([1,3,1,1]))
        perception_loss = creMSE(out_feat.reshape(out_feat.size(0),-1), \
                                target_feat.reshape(target_feat.size(0),-1))

        tloss =  0.5 * mseloss + 0.5 * maeloss \
            + perception_loss.to(device) * 0.002 + tvloss(fake_hr.cpu()).to(device) * 2e-8
        # accmu
        tloss = tloss/cfg.accmu
        tloss.backward(retain_graph=False)

        if((i+1) % cfg.accmu == 0):
            optimizer.step()
            optimizer.zero_grad()
            l = "train: epoch %d,batch %d,mae loss %f,mse loss %f,per loss %f.\n" % \
                (e,i,maeloss.cpu().detach().numpy(),mseloss.cpu().detach().numpy(),perception_loss.detach().numpy())
            ## wlog(cfg.logn,l)

            train_bar.set_description(\
                desc=f"[{e}/{cfg.epoc_num}] Loss: {tloss.cpu().detach().numpy():.4f}")

    # chr k is over and eval.
    ## t.save(mdl,"msave/Model"+ "-" + str(e)+".pt")
    # t.save(mdl, cfg.out_dir + "Model"+ "-" + str(e)+".pt")

    mdl.eval()
    ssim_epoc = 0
    t.cuda.empty_cache()
    valid_bar = tqdm(val_loader)
    with t.no_grad():

        for j,(lrdata,hrdata) in enumerate(valid_bar):
            lrdata = lrdata.to(device).unsqueeze(dim = 1).to(t.float32)
            hrdata = hrdata.to(device).unsqueeze(dim = 1).to(t.float32)
            fa_lr = mdl(lrdata)

            ################################################## by HiHiC ###
            mseloss = creMSE(fa_lr,hrdata) ################################
            maeloss = creMAE(fa_lr,hrdata)
            out_feat = loss_network(fa_lr.cpu().repeat([1,3,1,1]))
            target_feat = loss_network(hrdata.cpu().repeat([1,3,1,1]))
            perception_loss = creMSE(out_feat.reshape(out_feat.size(0),-1), \
                                    target_feat.reshape(target_feat.size(0),-1))

            vloss =  0.5 * mseloss + 0.5 * maeloss \
                + perception_loss.to(device) * 0.002 + tvloss(fa_lr.cpu()).to(device) * 2e-8
            # accmu
            vloss = vloss/cfg.accmu #######################################
            ###############################################################

            hrdata = t.minimum(hrdata,t.tensor(255).to(device)) / 255
            fa_lr = t.minimum(fa_lr,t.tensor(255).to(device)) / 255

            batch_mse = ((fa_lr - hrdata) ** 2).mean()
            batch_mae = (abs(fa_lr - hrdata)).mean()
            batch_ssim = ssim(fa_lr, hrdata)
            psnr = 10 * log10(1 / batch_mse)
            ssim_epoc += batch_ssim
 
            l = "test: epoch %d,ssim %f,psnr %f.\n" % (e,batch_ssim,psnr)
            ## wlog(cfg.logn,l)

            valid_bar.set_description(desc=f"[Predicting in Test set]\
                 PSNR: {psnr:.4f} dB SSIM: {batch_ssim:.4f}")

    now_ssim = ssim_epoc / j
    if now_ssim > best_ssim:
        best_ssim = now_ssim
        print(f'Now, Best ssim is {best_ssim:.6f}')
        ## t.save(mdl,"predictm/Model-Best"+ "-" + str(e)+".pt")
        # t.save(mdl, cfg.out_dir + "Model-Best"+ "-" + str(e)+".pt")
        
    # lr d
    lrdecay.step()
    
    
    if e: ############################################################################################################# Added by HiHiC ####
        sec = time.time()-start ###########################################################################################################
        times = str(datetime.timedelta(seconds=sec))
        short = times.split(".")[0].replace(':','.')		
        train_epoch.append(e)
        train_time.append(short)        
        train_loss.append(f"{tloss:.10f}")
        valid_loss.append(f"{vloss:.10f}")
        ckpt_file = f"{str(e).zfill(5)}_{short}_{vloss:.10f}"
        t.save(mdl.state_dict(), os.path.join(args.output_model_dir, ckpt_file)) ###########################################################
        np.save(os.path.join(args.loss_log_dir, f'train_loss_{args.model}'), [train_epoch, train_time, train_loss, valid_loss]) ############
