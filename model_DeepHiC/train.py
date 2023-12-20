import os
import time
# import visdom 
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.deephic import Generator, Discriminator
from models.loss import GeneratorLoss
from models.ssim import ssim
from math import log10

from all_parser import root_dir


######################################################## Added by HiHiC ######
##############################################################################

import datetime, argparse

parser = argparse.ArgumentParser(description= 'DeepHiC training process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True,
                      help='HiHiC directory')
required.add_argument('--model', type=str, metavar='DeepHiC', required=True,
                      help='model name')
required.add_argument('--epoch', type=int, default=128, metavar='[2]', required=True,
                      help='training epoch (default: 128)')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True,
                      help='input batch size for training (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, 
                      help='GPU ID for training (defalut: 0)')
required.add_argument('--output_model_dir', type=str, default='./checkpoints_DeepHiC', metavar='[5]', required=True,
                      help='directory path of training model (default: HiHiC/checkpoints_DeepHiC/)')
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
train_time = []

os.makedirs(args.loss_log_dir, exist_ok=True)
torch.autograd.set_detect_anomaly(True)
##############################################################################
##############################################################################

cs = np.column_stack

# data_dir: directory storing processed data
data_dir = os.path.join(root_dir, 'data')

# out_dir: directory storing checkpoint files
out_dir = args.output_model_dir
os.makedirs(out_dir, exist_ok=True)

datestr = time.strftime('%m_%d_%H_%M')
# visdom_str=time.strftime('%m%d')

# resos = '10kb40kb'
# chunk = 40
# stride = 40
# bound = 201
# pool = 'nonpool'

upscale = 1
num_epochs = args.epoch
batch_size = args.batch_size

# whether using GPU for training
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

# prepare training dataset
# train_file = os.path.join(data_dir, f'deephic_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_train.npz')
# train = np.load(train_file)
data_all = [np.load(os.path.join(args.train_data_dir, fname), allow_pickle=True) for fname in os.listdir(args.train_data_dir)] ### Added by HiHiC ##
train = {} #########################################################################################################################################
for data in data_all: ##############################################################################################################################
    [train.update({k: v}) for k, v in data.items()] ################################################################################################

train_data = torch.tensor(train['data'], dtype=torch.float)
train_target = torch.tensor(train['target'], dtype=torch.float)
train_inds = torch.tensor(train['inds'], dtype=torch.long) 

train_set = TensorDataset(train_data, train_target, train_inds)

# prepare valid dataset
# valid_file = os.path.join(data_dir, f'deephic_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_valid.npz')
# valid = np.load(valid_file)
data_all = [np.load(os.path.join(args.valid_data_dir, fname), allow_pickle=True) for fname in os.listdir(args.valid_data_dir)] ### Added by HiHiC ##
valid = {} #########################################################################################################################################
for data in data_all: ##############################################################################################################################
    [train.update({k: v}) for k, v in data.items()] ################################################################################################

valid_data = torch.tensor(valid['data'], dtype=torch.float)
valid_target = torch.tensor(valid['target'], dtype=torch.float)
valid_inds = torch.tensor(valid['inds'], dtype=torch.long) 

valid_set = TensorDataset(valid_data, valid_target, valid_inds)

# DataLoader for batched training
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)

# load network
netG = Generator(upscale, in_channel=1, resblock_num=5).to(device)
netD = Discriminator(in_channel=1).to(device)

# loss function
criterionG = GeneratorLoss().to(device)
criterionD = torch.nn.BCELoss().to(device)

# optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.0001)
optimizerD = optim.Adam(netD.parameters(), lr=0.0001)

# vis = visdom.Visdom(env=f'{visdom_str}-deephic')

best_ssim = 0
for epoch in range(1, num_epochs+1):
    run_result = {'nsamples': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    train_bar = tqdm(train_loader)
    for data, target, _ in train_bar:
        batch_size = data.size(0)
        run_result['nsamples'] += batch_size
        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = target.to(device)
        z = data.to(device)
        fake_img = netG(z)

        ######### Train discriminator #########
        netD.zero_grad()
        real_out = netD(real_img)
        fake_out = netD(fake_img)
        d_loss_real = criterionD(real_out, torch.ones_like(real_out))
        d_loss_fake = criterionD(fake_out, torch.zeros_like(fake_out))
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ######### Train generator #########
        netG.zero_grad()
        fake_out = netD(fake_img) ############################################################# Editted by HiHiC #####
        g_loss = criterionG(fake_out.mean(), fake_img, real_img) #####################################################
        g_loss.backward()
        ##############################################################################################################
        # optimizerD.step() #################################### (refer to GitHub issue; issue about training step) ##
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size
        run_result['d_loss'] += d_loss.item() * batch_size
        run_result['d_score'] += real_out.mean().item() * batch_size
        run_result['g_score'] += fake_out.mean().item() * batch_size

        train_bar.set_description(desc=f"[{epoch}/{num_epochs}] Loss_D: {run_result['d_loss']/run_result['nsamples']:.4f} Loss_G: {run_result['g_loss']/run_result['nsamples']:.4f} D(x): {run_result['d_score']/run_result['nsamples']:.4f} D(G(z)): {run_result['g_score']/run_result['nsamples']:.4f}")
    train_gloss = run_result['g_loss']/run_result['nsamples']
    train_dloss = run_result['d_loss']/run_result['nsamples']
    train_dscore = run_result['d_score']/run_result['nsamples']
    train_gscore = run_result['g_score']/run_result['nsamples']

    valid_result = {'g_loss': 0, 'd_loss': 0, 'g_score': 0, 'd_score': 0, 
                    'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    netG.eval()
    netD.eval()
    valid_bar = tqdm(valid_loader)
    with torch.no_grad():
        for val_lr, val_hr, inds in valid_bar:
            batch_size = val_lr.size(0)
            valid_result['nsamples'] += batch_size
            lr = val_lr.to(device)
            hr = val_hr.to(device)
            sr = netG(lr)

            sr_out = netD(sr)
            hr_out = netD(hr)
            d_loss_real = criterionD(hr_out, torch.ones_like(hr_out))
            d_loss_fake = criterionD(sr_out, torch.zeros_like(sr_out))
            d_loss = d_loss_real + d_loss_fake
            g_loss = criterionG(sr_out.mean(), sr, hr)
            
            valid_result['g_loss'] += g_loss.item() * batch_size
            valid_result['d_loss'] += d_loss.item() * batch_size
            valid_result['g_score'] += sr_out.mean().item() * batch_size
            valid_result['d_score'] += hr_out.mean().item() * batch_size

            batch_mse = ((sr - hr) ** 2).mean()
            valid_result['mse'] += batch_mse * batch_size
            batch_ssim = ssim(sr, hr)
            valid_result['ssims'] += batch_ssim * batch_size
            valid_result['psnr'] = 10 * log10(1/(valid_result['mse']/valid_result['nsamples']))
            valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
            valid_bar.set_description(desc=f"[Predicting in Test set] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")

    valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
    valid_dloss = valid_result['d_loss'] / valid_result['nsamples']
    valid_gscore = valid_result['g_score'] / valid_result['nsamples']
    valid_dscore = valid_result['d_score'] / valid_result['nsamples']
    now_ssim = valid_result['ssim'].item()
    
    # if epoch == 1:
    #     vis_dloss = vis.line(X=cs((epoch, epoch)), Y=cs((train_dloss, valid_dloss)), opts=dict(title='Discriminator Loss', legend=['Train', 'Test']))
    #     vis_gloss = vis.line(X=cs((epoch, epoch)), Y=cs((train_gloss, valid_gloss)), opts=dict(title='Generator Loss', legend=['Train', 'Test']))
    #     vis_dscore = vis.line(X=cs((epoch, epoch)), Y=cs((train_dscore, valid_dscore)), opts=dict(title='Discriminator Score of true images', legend=['Train', 'Test']))
    #     vis_gscore = vis.line(X=cs((epoch, epoch)), Y=cs((train_gscore, valid_gscore)), opts=dict(title='Generator Score of fake images', legend=['Train', 'Test']))
    #     vis_ssim = vis.line([now_ssim], X=[epoch], opts=dict(title='SSIM scores in test dataset'))
    # else:
    #     vis.line(X=cs((epoch, epoch)), Y=cs((train_dloss, valid_dloss)), update='append', win=vis_dloss, opts=dict(legend=['Train', 'Test']))
    #     vis.line(X=cs((epoch, epoch)), Y=cs((train_gloss, valid_gloss)), update='append', win=vis_gloss, opts=dict(legend=['Train', 'Test']))
    #     vis.line(X=cs((epoch, epoch)), Y=cs((train_dscore, valid_dscore)), update='append', win=vis_dscore, opts=dict(legend=['Train', 'Test']))
    #     vis.line(X=cs((epoch, epoch)), Y=cs((train_gscore, valid_gscore)), update='append', win=vis_gscore, opts=dict(legend=['Train', 'Test']))
    #     vis.line([now_ssim], X=[epoch], update='append', win=vis_ssim)

    if now_ssim > best_ssim:
        best_ssim = now_ssim
        print(f'Now, Best ssim is {best_ssim:.6f}')
        # best_ckpt_file = f'{datestr}_bestg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_deephic.pytorch' 
        best_ckpt_file = f'{datestr}_bestg_deephic.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, best_ckpt_file))
        
    if epoch%10 == 0: ########################################## Added by HiHiC ####
        sec = time.time()-start ####################################################
        times = str(datetime.timedelta(seconds=sec))
        short = times.split(".")[0].replace(':','.')
        train_epoch.append(epoch)
        train_time.append(short)        
        train_loss.append(f"{now_ssim:.2f}")
        ckpt_file = f"{str(epoch).zfill(5)}_{short}.pytorch" #######################
        torch.save(netG.state_dict(), os.path.join(out_dir, ckpt_file)) ############
    
final_ckpt_g = f'{datestr}_finalg_deephic.pytorch'
final_ckpt_d = f'{datestr}_finald_deephic.pytorch'

torch.save(netG.state_dict(), os.path.join(out_dir, final_ckpt_g))
torch.save(netD.state_dict(), os.path.join(out_dir, final_ckpt_d))

np.save(os.path.join(args.loss_log_dir, f'train_loss_{args.model}'), [train_epoch, train_time, train_loss]) ### Added by HiHiC ##