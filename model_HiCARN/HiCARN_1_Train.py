import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from Models.HiCARN_1 import Generator
from Models.HiCARN_1_Loss import GeneratorLoss
from Utils.SSIM import ssim
from math import log10


##################################################################

import datetime

ROOT_DIR = './'
OUT_DIR = os.path.join(ROOT_DIR, 'checkpoints_HiCARN1')
TRAIN_FILE = '/data/HiHiC-main/data_HiCARN/Train_and_Validation/train_ratio16.npz'
VALID_FILE = '/data/HiHiC-main/data_HiCARN/Train_and_Validation/train_ratio16.npz'
LOSS_LOG = 'train_loss_HiCARN1.npy'
NUM_EPOCHS = 500
BATCH_SIZE = 64

start = time.time()

train_epoch = [] 
train_loss = []
train_time = []

##################################################################

cs = np.column_stack

# root_dir = ROOT_DIR

def adjust_learning_rate(epoch):
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr


# data_dir: directory storing processed data
# data_dir = DATA_DIR

# out_dir: directory storing checkpoint files
out_dir = OUT_DIR
os.makedirs(out_dir, exist_ok=True)

datestr = time.strftime('%m_%d_%H_%M')
visdom_str = time.strftime('%m%d')

num_epochs = NUM_EPOCHS
batch_size = BATCH_SIZE

# whether using GPU for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("CUDA available? ", torch.cuda.is_available())
print("Device being used: ", device)

# prepare training dataset
train_file = TRAIN_FILE
train = np.load(train_file, allow_pickle=True)

train_data = torch.tensor(train['data'], dtype=torch.float)
train_target = torch.tensor(train['target'], dtype=torch.float)
train_inds = torch.tensor(train['inds'], dtype=torch.long)

train_set = TensorDataset(train_data, train_target, train_inds)

# prepare valid dataset
valid_file = VALID_FILE
valid = np.load(valid_file, allow_pickle=True)

valid_data = torch.tensor(valid['data'], dtype=torch.float)
valid_target = torch.tensor(valid['target'], dtype=torch.float)
valid_inds = torch.tensor(valid['inds'], dtype=torch.long)

valid_set = TensorDataset(valid_data, valid_target, valid_inds)

# DataLoader for batched training
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)

# load network
netG = Generator(num_channels=64).to(device)

# loss function
criterionG = GeneratorLoss().to(device)

# optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)

ssim_scores = []
psnr_scores = []
mse_scores = []
mae_scores = []

best_ssim = 0
for epoch in range(1, num_epochs + 1):
    run_result = {'nsamples': 0, 'g_loss': 0, 'g_score': 0}

    alr = adjust_learning_rate(epoch)
    optimizerG = optim.Adam(netG.parameters(), lr=alr)

    for p in netG.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    torch.cuda.empty_cache()

    netG.train()
    train_bar = tqdm(train_loader)
    for data, target, _ in train_bar:
        batch_size = data.size(0)
        run_result['nsamples'] += batch_size

        real_img = target.to(device)
        z = data.to(device)
        fake_img = netG(z)

        ######### Train generator #########
        netG.zero_grad()
        g_loss = criterionG(fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size

        train_bar.set_description(
            desc=f"[{epoch}/{num_epochs}] Loss_G: {run_result['g_loss'] / run_result['nsamples']:.4f}")
    train_gloss = run_result['g_loss'] / run_result['nsamples']
    train_gscore = run_result['g_score'] / run_result['nsamples']

    valid_result = {'g_loss': 0,
                    'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    netG.eval()

    batch_ssims = []
    batch_mses = []
    batch_psnrs = []
    batch_maes = []

    valid_bar = tqdm(valid_loader)
    with torch.no_grad():
        for val_lr, val_hr, inds in valid_bar:
            batch_size = val_lr.size(0)
            valid_result['nsamples'] += batch_size
            lr = val_lr.to(device)
            hr = val_hr.to(device)
            sr = netG(lr)

            sr_out = sr
            hr_out = hr
            g_loss = criterionG(sr, hr)

            valid_result['g_loss'] += g_loss.item() * batch_size

            batch_mse = ((sr - hr) ** 2).mean()
            batch_mae = (abs(sr - hr)).mean()
            valid_result['mse'] += batch_mse * batch_size
            batch_ssim = ssim(sr, hr)
            valid_result['ssims'] += batch_ssim * batch_size
            valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
            valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
            valid_bar.set_description(
                desc=f"[Predicting in Test set] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")

            batch_ssims.append(valid_result['ssim'])
            batch_psnrs.append(valid_result['psnr'])
            batch_mses.append(batch_mse)
            batch_maes.append(batch_mae)

    ssim_scores.append((sum(batch_ssims) / len(batch_ssims)))
    psnr_scores.append((sum(batch_psnrs) / len(batch_psnrs)))
    mse_scores.append((sum(batch_mses) / len(batch_mses)))
    mae_scores.append((sum(batch_maes) / len(batch_maes)))

    valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
    now_ssim = valid_result['ssim'].item()

    if now_ssim > best_ssim:
        best_ssim = now_ssim
        print(f'Now, Best ssim is {best_ssim:.6f}')
        best_ckpt_file = f'{datestr}_bestg.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, best_ckpt_file))
    
    ##################################################################
        
    if epoch%10 == 0:
        sec = time.time()-start
        times = str(datetime.timedelta(seconds=sec))
        short = times.split(".")[0].replace(':','.')
            
        train_epoch.append(epoch)
        train_time.append(short)        
        train_loss.append(f"{now_ssim:.2f}")
        
        ckpt_file = f"{str(epoch).zfill(5)}_{short}.pytorch"
        torch.save(netG.state_dict(), os.path.join(out_dir, ckpt_file))
    
    ##################################################################
        
final_ckpt_g = f'{datestr}_finalg.pytorch'

######### Uncomment to track scores across epochs #########
# ssim_scores = ssim_scores.cpu()
# psnr_scores = psnr_scores.cpu()
# mse_scores = mse_scores.cpu()
# mae_scores = mae_scores.cpu()

# ssim_scores = np.array(ssim_scores)
# psnr_scores = np.array(psnr_scores)
# mse_scores = np.array(mse_scores)
# mae_scores = np.array(mae_scores)
#
# np.savetxt(f'valid_ssim_scores_{name}', X=ssim_scores, delimiter=',')
# np.savetxt(f'valid_psnr_scores_{name}', X=psnr_scores, delimiter=',')
# np.savetxt(f'valid_mse_scores_{name}', X=mse_scores, delimiter=',')
# np.savetxt(f'valid_mae_scores_{name}', X=mae_scores, delimiter=',')

torch.save(netG.state_dict(), os.path.join(out_dir, final_ckpt_g))


##################################################################

np.save(LOSS_LOG, [train_epoch, train_time, train_loss])

##################################################################