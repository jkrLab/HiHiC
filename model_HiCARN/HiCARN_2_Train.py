import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from Models.HiCARN_2 import Generator, Discriminator
from Models.HiCARN_2_Loss import GeneratorLoss
from Utils.SSIM import ssim
from math import log10


################################################## Added by HiHiC ######
import datetime, argparse, random ######################################

seed = 13
random.seed(seed)  # Python 기본 랜덤 시드
np.random.seed(seed)  # NumPy 랜덤 시드
torch.manual_seed(seed)  # CPU 랜덤 시드

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 시드 적용

parser = argparse.ArgumentParser(description='HiCARN2 training process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True,
                      help='HiHiC directory')
required.add_argument('--model', type=str, metavar='HiCARN2', required=True,
                      help='model name')
required.add_argument('--epoch', type=int, default=128, metavar='[2]', required=True,
                      help='training epoch (default: 128)')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True,
                      help='input batch size for training (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, 
                      help='GPU ID for training (defalut: 0)')
required.add_argument('--output_model_dir', type=str, default='./checkpoints_HiCARN2', metavar='[5]', required=True,
                      help='directory path of training model (default: HiHiC/checkpoints_HiCARN2/)')
required.add_argument('--loss_log_dir', type=str, default='./log', metavar='[6]', required=True,
                      help='directory path of training log (default: HiHiC/log/)')
required.add_argument('--train_data_dir', type=str, metavar='[7]', required=True,
                      help='directory path of training data')
optional.add_argument('--valid_data_dir', type=str, metavar='[8]',
                      help="directory path of validation data, but HiCPlus doesn't need")
args = parser.parse_args()


start = time.time()

train_epoch = [] 
train_loss = []
valid_loss = []
train_time = []

def calculate_initial_loss(generator, discriminator, data_loader, criterion, device):
    generator.eval()
    discriminator.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for data, target, _ in data_loader:
            batch_size = data.size(0)
            data, target = data.to(device), target.to(device)

            # Generator의 출력
            fake_img = generator(data)

            # Discriminator를 통해 out_labels 계산
            out_labels = discriminator(fake_img)

            # 손실 계산
            loss = criterion(out_labels, fake_img, target)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples

os.makedirs(args.loss_log_dir, exist_ok=True) ##########################
########################################################################

cs = np.column_stack

root_dir = args.root_dir

def adjust_learning_rate(epoch):
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr


# data_dir: directory storing processed data
# data_dir = DATA_DIR

# out_dir: directory storing checkpoint files
out_dir = args.output_model_dir
os.makedirs(out_dir, exist_ok=True)

datestr = time.strftime('%m_%d_%H_%M')
# visdom_str = time.strftime('%m%d')

# resos = '10kb40kb'
# chunk = 40
# stride = 40
# bound = 201
# pool = 'nonpool'
# name = 'HiCARN_1'

num_epochs = args.epoch
batch_size = args.batch_size

# whether using GPU for training
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
print("CUDA available? ", torch.cuda.is_available())
print("Device being used: ", device)

# prepare training dataset
# train_file = os.path.join(data_dir, f'hicarn_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_train.npz')
# train = np.load(train_file)
data_all = [np.load(os.path.join(args.train_data_dir, fname), allow_pickle=True) for fname in os.listdir(args.train_data_dir)] ### Added by HiHiC ##
train = {'data': [], 'target': [], 'inds': []} #####################################################################################################
for data in data_all:
    for k, v in data.items():
        if k in train: 
            train[k].append(v) #####################################################################################################################
train = {k: np.concatenate(v, axis=0) for k, v in train.items()} ###################################################################################

train_data = torch.tensor(train['data'], dtype=torch.float)
train_target = torch.tensor(train['target'], dtype=torch.float)
train_inds = torch.tensor(train['inds'], dtype=torch.long)

train_set = TensorDataset(train_data, train_target, train_inds)

# prepare valid dataset
# valid_file = os.path.join(data_dir, f'hicarn_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_valid.npz')
# valid = np.load(valid_file)
data_all = [np.load(os.path.join(args.valid_data_dir, fname), allow_pickle=True) for fname in os.listdir(args.valid_data_dir)] ### Added by HiHiC ##
valid = {'data': [], 'target': [], 'inds': []} #####################################################################################################
for data in data_all:
    for k, v in data.items():
        if k in valid: 
            valid[k].append(v) #####################################################################################################################
valid = {k: np.concatenate(v, axis=0) for k, v in valid.items()} ###################################################################################

valid_data = torch.tensor(valid['data'], dtype=torch.float)
valid_target = torch.tensor(valid['target'], dtype=torch.float)
valid_inds = torch.tensor(valid['inds'], dtype=torch.long)

valid_set = TensorDataset(valid_data, valid_target, valid_inds)

# DataLoader for batched training
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)

# load network
netG = Generator(num_channels=64).to(device)
netD = Discriminator().to(device)

# loss function
criterionG = GeneratorLoss().to(device)
criterionD = torch.nn.BCELoss().to(device)

# optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)
optimizerD = optim.Adam(netD.parameters(), lr=0.0003)

ssim_scores = []
psnr_scores = []
mse_scores = []
mae_scores = []

######################################################################################################## Added by HiHiC ####
initial_train_loss = calculate_initial_loss(netG, netD, train_loader, criterionG, device) ##################################
initial_valid_loss = calculate_initial_loss(netG, netD, valid_loader, criterionG, device) 
train_epoch.append(int(0))
train_time.append("0.00.00")        
train_loss.append(f"{initial_train_loss:.10f}") 
valid_loss.append(f"{initial_valid_loss:.10f}") ############################################################################
np.save(os.path.join(args.loss_log_dir, f'train_loss_{args.model}'), [train_epoch, train_time, train_loss, valid_loss]) ####

best_ssim = 0
for epoch in range(1, num_epochs + 1):
    run_result = {'nsamples': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    alr = adjust_learning_rate(epoch)
    optimizerG = optim.Adam(netG.parameters(), lr=alr)
    optimizerD = optim.Adam(netD.parameters(), lr=alr)

    for p in netG.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    torch.cuda.empty_cache()

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
        g_loss = criterionG(fake_out.mean(), fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size
        run_result['d_loss'] += d_loss.item() * batch_size
        run_result['d_score'] += real_out.mean().item() * batch_size
        run_result['g_score'] += fake_out.mean().item() * batch_size

        train_bar.set_description(
            desc=f"[{epoch}/{num_epochs}] "
                 f"Loss_D: {run_result['d_loss'] / run_result['nsamples']:.4f} "
                 f"Loss_G: {run_result['g_loss'] / run_result['nsamples']:.4f} "
                 f"D(x): {run_result['d_score'] / run_result['nsamples']:.4f} "
                 f"D(G(z)): {run_result['g_score'] / run_result['nsamples']:.4f}")

    train_gloss = run_result['g_loss'] / run_result['nsamples']
    train_dloss = run_result['d_loss'] / run_result['nsamples']
    train_dscore = run_result['d_score'] / run_result['nsamples']
    train_gscore = run_result['g_score'] / run_result['nsamples']

    valid_result = {'g_loss': 0, 'd_loss': 0, 'g_score': 0, 'd_score': 0,
                    'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    netG.eval()
    netD.eval()

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
    valid_dloss = valid_result['d_loss'] / valid_result['nsamples']
    valid_gscore = valid_result['g_score'] / valid_result['nsamples']
    valid_dscore = valid_result['d_score'] / valid_result['nsamples']
    now_ssim = valid_result['ssim'].item()

    if now_ssim > best_ssim:
        best_ssim = now_ssim
        print(f'Now, Best ssim is {best_ssim:.6f}')
        best_ckpt_file = f'{datestr}_bestg.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, best_ckpt_file))

    if epoch: ##################################################################################################### Added by HiHiC #####
        sec = time.time()-start ########################################################################################################
        times = str(datetime.timedelta(seconds=sec))
        short = times.split(".")[0].replace(':','.') 
        train_epoch.append(epoch) 
        train_time.append(short)       
        train_loss.append(f"{train_gloss:.10f}") 
        valid_loss.append(f"{valid_gloss:.10f}") 
        ckpt_file = f"{str(epoch).zfill(5)}_{short}_{valid_gloss:.10f}" 
        np.save(os.path.join(args.loss_log_dir, f'train_loss_{args.model}'), [train_epoch, train_time, train_loss, valid_loss]) #########
        torch.save(netG.state_dict(), os.path.join(out_dir, ckpt_file)) #################################################################    
        
# final_ckpt_g = f'{datestr}_finalg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'        
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