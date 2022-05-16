import random, os
import wandb
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import argparse
import importlib
import numpy as np
import pickle
import torch.utils.data
from skimage.util import random_noise
from VDPLayers_VarCoVar import VDP_Flatten, VDP_Conv2D, VDP_Relu, VDP_Maxpool, VDP_FullyConnected, VDP_Softmax, VDP_BatchNorm2D, VDP_AdaptiveAvgPool2d
import logging
import torchattacks
from pytorch_lightning import seed_everything


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser(prog="eVI")

# Training Parameters
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--batch_size_train', type=int, default=32, help='Batch size for training')
# Testing Parameters
parser.add_argument('--testing', type=bool, default=False, help='Run inference using the trained model checkpoint')
parser.add_argument('--batch_size_val', type=int, default=16, help='Batch size for validation')
parser.add_argument('--batch_size_test', type=int, default=1, help='Batch size for testing, can only be 1')
#Adding Noise/Adversarial attack
parser.add_argument('--add_noise', type=bool, default=False, help='Addition of noise during testing')
parser.add_argument('--adv_attack', type=bool, default=False, help='Adding adversarial attack during testing')
parser.add_argument('--adv_trgt', type=int, default=313, help='Adding adversarial attack Target')
# Load Model
parser.add_argument('--load_model', type=str, default='', help='Path to a previously trained model checkpoint')
parser.add_argument('--data_path', type=str, default='', help='Path to save dataset data')
# Loss Function Parameters
parser.add_argument('--tau_conv1', type=float, default=0.001, help='KL Weight Term') 
parser.add_argument('--tau_b2', type=float, default=0.01, help='KL Weight Term') 
parser.add_argument('--tau_b3', type=float, default=0.01, help='KL Weight Term') 
parser.add_argument('--tau_b4', type=float, default=0.01, help='KL Weight Term') 
parser.add_argument('--tau_fc', type=float, default=0.001, help='KL Weight Term') 
parser.add_argument('--clamp', type=float, default=1000, help='Clamping')
parser.add_argument('--var_sup', type=float, default=0.001, help='Loss Variance Bias')
# Learning Rate Parameters
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--wd', type=float, default=0.00005, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum used for Optimizer')

parser.add_argument('--output_size', type=int, default=100, help='Size of the output')

######################################################################################################################################
#VDP Model
############################################################
class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, stride=1, input_flag=False):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        self.stride = stride
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
#         self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
#         self.bn11 = nn.BatchNorm2d(out_channels)
        self.conv11 = VDP_Conv2D(in_channels, out_channels, kernel_size=1, stride=2)
        self.bn11 = VDP_BatchNorm2D(out_channels)
        self.relu_ = VDP_Relu()
       
    def forward(self, x, sig):
        identity = x
        sigma = sig
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.stride != 1:
            identity, sigma = self.conv11(identity, sigma)
            identity, sigma = self.bn11(identity, sigma)

        x += identity
        x, sigma = self.relu_(x, sigma)
        return x, sigma


class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.output_size = num_classes
        self.conv1 = VDP_Conv2D(image_channels, 64, kernel_size=7, stride=2, padding=3, input_flag=True)
        self.bn1 = VDP_BatchNorm2D(64)
        self.relu = VDP_Relu()
        self.maxpool = VDP_Maxpool(kernel_size=3, stride=2, padding=1)
#         self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block11 = block(18, 64, 64, stride=1)
        self.block12 = block(18, 64, 64, stride=1)
        self.block21 = block(18, 64, 128, stride=2)
        self.block22 = block(18, 128, 128, stride=1)
        self.block31 = block(18, 128, 256, stride=2)
        self.block32 = block(18, 256, 256, stride=1)
        self.block41 = block(18, 256, 512, stride=2)
        self.block42 = block(18, 512, 512, stride=1)

        self.avgpool = VDP_Avgpool2d()
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = VDP_FullyConnected(512 * self.expansion, num_classes)
        self.flatten = VDP_Flatten()
        self.softmax = VDP_Softmax()

    def forward(self, x):
        x, sigma = self.conv1(x)
        x, sigma = self.bn1(x, sigma)
        x, sigma = self.relu(x, sigma)
        x, sigma = self.maxpool(x, sigma)
        
        x, sigma = self.block11(x, sigma)
        x, sigma = self.block12(x, sigma)
        
        x, sigma = self.block21(x, sigma)
        x, sigma = self.block22(x, sigma)
        
        x, sigma = self.block31(x, sigma)
        x, sigma = self.block32(x, sigma)
        
        x, sigma = self.block41(x, sigma)
        x, sigma = self.block42(x, sigma)
    
        x, sigma = self.avgpool(x, sigma)
        x, sigma = self.flatten(x, sigma)
        
        x, sigma = self.fc(x, sigma)
        x, sigma = self.softmax(x, sigma)

        return x, sigma
    
    def nll_gaussian(self, y_pred_mean, y_pred_sd, y_test):
        NS = torch.diag(torch.ones(self.output_size, device=y_pred_sd.device) * torch.tensor(
            0.001, device=y_pred_sd.device))
        y_pred_sd_inv = torch.inverse(y_pred_sd + NS)
        mu_ = y_pred_mean - y_test
        mu_sigma = torch.bmm(mu_.unsqueeze(1), y_pred_sd_inv)
        ms = ((torch.bmm(mu_sigma, mu_.unsqueeze(2)).squeeze(1)).mean() +
              ((torch.slogdet(y_pred_sd + NS)[1]).unsqueeze(1)).mean()).mean()
        return ms   

    
    def batch_loss(self, output_mean, output_sigma, target):
        output_sigma_clamp = torch.clamp(output_sigma,-1000,1000)
        neg_log_likelihood = self.nll_gaussian(output_mean, output_sigma_clamp, target)

        loss_value = 0.05*neg_log_likelihood + (self.tau_conv1*self.conv1.kl_loss_term() +
                                               self.tau_b2*self.block21.conv11.kl_loss_term() + 
                                               self.tau_b3*self.block31.conv11.kl_loss_term() +
                                               self.tau_b4*self.block41.conv11.kl_loss_term() +
                                               self.tau_fc*self.fc.kl_loss_term())        
        return loss_value

####################################################################################################

class SelectOutput(nn.Module):
    def __init__(self):
        super(SelectOutput, self).__init__()

    def forward(self,x):
        out = x[0]
        return out

########################################################
    
def train(args, model, optimizer, train_loader, epoch):
    model.train()
    
    train_losses = []
    train_counter = []
    train_acc = 0
    total_num = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # transfer data to the current GPU device
        data, target = data.to(args.devices), target.to(args.devices)
        optimizer.zero_grad()
        mu_o, sigma_o = model(data)
        labels = nn.functional.one_hot(target, args.output_size)
        loss = model.batch_loss(mu_o, sigma_o, labels)        
        _, pred = mu_o.max(1, keepdim=True)

        train_acc += pred.eq(target.view_as(pred)).sum().item()
        total_num += len(target)        
        loss.backward()
#         nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()
#         print(model.conv1.sigma_weight.grad)
#         print(model.fc.sigma_weight.grad)
                
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*128) + ((epoch-1)*len(train_loader.dataset)))
    acc = train_acc / total_num
    print('Train Accuracy: ', acc)    
    
    torch.save(model.state_dict(), '../../Results/checkpoints/CIFAR100_VDP_Res18_model_251121_1500.pth')
    torch.save(optimizer.state_dict(), '../../Results/checkpoints/CIFAR100_VDP_Res18_optimizer_251121_1500.pth')
    
    return loss, acc

########################################################

def validation(args, model, valid_loader):
    
    model.eval()
    model.zero_grad()
    val_acc = 0
    total_num = 0
    for idx, (data, targets) in enumerate(valid_loader):
        data, targets = data.to(args.devices), targets.to(args.devices)

        mu_y_out, sigma_y_out = model.forward(data)

        _, pred = mu_y_out.max(1, keepdim=True)

        val_acc += pred.eq(targets.view_as(pred)).sum().item()
        total_num += len(targets)
      
    acc = val_acc / total_num
    print('Validation Accuracy: ', acc)
    
    return acc

########################################################

def test(args, model, test_loader, noise_std):
    print('Starting Test Phase')
#     model.eval()
#     model.zero_grad()
    test_acc = 0
    total = 0
    actual, target = list(), list()
    noise_data = list()
    corrmu, imu = list(),list()
    corrvar, ivar = list(), list()
    prediction, variance = list(), list()
    corr_noise, corr_target = list(), list()
    i_noise, i_target = list(), list()
    
    for idx, (data, targets) in enumerate(test_loader):
        
        noise = random_noise(data, mode='gaussian', mean=0, var=(noise_std) ** 2, clip=True)
        noisy_img = torch.from_numpy(noise)

        mu_y_out, sigma_y_out = model.forward(noisy_img.float())

        _, pred = mu_y_out.max(1, keepdim=True)

        test_acc += pred.eq(targets.view_as(pred)).sum().item()
        total += len(targets)
        
        correctpred = pred.eq(targets.view_as(pred))
        actual.append(data.numpy())
        noise_data.append(noise)
        target.append(targets.reshape((len(targets), 1)).numpy())
        prediction.append(mu_y_out.detach().numpy())
        variance.append(sigma_y_out.detach().numpy())

        if correctpred == True:
            corrmu.append(mu_y_out.detach().numpy())
            corrvar.append(sigma_y_out.detach().numpy())
            corr_noise.append(noise)
            corr_target.append(targets.reshape((len(targets), 1)).numpy())
        else:
            imu.append(mu_y_out.detach().numpy())
            ivar.append(sigma_y_out.detach().numpy())
            i_noise.append(noise)
            i_target.append(targets.reshape((len(targets), 1)).numpy())
      
    acc = 100*test_acc / total
    print('Noise_std: ', noise_std)
    print('Test Accuracy: ', acc)

    actual = np.vstack(actual)
    noise_data = np.vstack(noise_data)
    target = np.vstack(target)
    prediction = np.vstack(prediction)
    variance = np.vstack(variance)
    corrmu = np.vstack(corrmu)
    imu = np.vstack(imu)
    corrvar = np.vstack(corrvar)
    ivar = np.vstack(ivar)
    corr_noise = np.vstack(corr_noise)
    corr_target = np.vstack(corr_target)
    i_noise = np.vstack(i_noise)
    i_target = np.vstack(i_target)
    
    print('shape of actual :', noise_data.shape)
    
    snr = 10 * np.log10(np.squeeze((np.sum(np.square(actual), (1, 2, 3))) / (np.sum(np.square(actual - noise_data),
                                                                                    (1, 2, 3)))))
    mean_snr = np.mean(snr)    
    print('mean_snr :', mean_snr)
    
    with open('../../Results/withNoise/CIFAR100_Res18/CIFAR100_Res18_GN_Results_251121_1500_{}'.format(noise_std), 'wb') as pf:
        print('saving')
        pickle.dump([actual, noise_data, target, prediction, variance, corrmu, imu, corrvar, ivar, 
                     corr_noise, i_noise, corr_target, i_target], pf)
        pf.close()

########################################################

def test_adv(args, model, test_loader, attack, eps):
    print('Starting Test Phase for Adv Attack')
    test_acc = 0
    total = 0
    
    actual = list()
    adv_images, target = list(), list()
    corrmu, imu = list(),list()
    corrvar, ivar = list(), list()
    prediction, variance = list(), list()
    corr_adv, corr_target = list(), list()
    i_adv, i_target = list(), list()
    ctr = 0
    
    attack.set_mode_targeted(target_map_function=None)
    for data, targets in test_loader:
        adv_image = attack(data, torch.tensor([args.adv_trgt]))     
        targets = targets.to(args.devices)
        mu_y_out, sigma_y_out = model(adv_image)

        _, pred = mu_y_out.max(1, keepdim=True)

        test_acc += pred.eq(targets.view_as(pred)).sum().item()
        total += len(targets)
        ctr += 1

        adv_image = adv_image.cpu().numpy()
        correctpred = pred.eq(targets.view_as(pred))
        prediction.append(mu_y_out.detach().cpu().numpy())
        variance.append(sigma_y_out.detach().cpu().numpy())
        adv_images.append(adv_image)
        target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        actual.append(data.cpu().numpy())

        if correctpred == True:
            corrmu.append(mu_y_out.detach().cpu().numpy())
            corrvar.append(sigma_y_out.detach().cpu().numpy())
            corr_adv.append(adv_image)
            corr_target.append(targets.reshape((len(targets), 1)).cpu().numpy())
        else:
            imu.append(mu_y_out.detach().cpu().numpy())
            ivar.append(sigma_y_out.detach().cpu().numpy())
            i_adv.append(adv_image)
            i_target.append(targets.reshape((len(targets), 1)).cpu().numpy())

        if ctr>199:
            print(ctr, 'images for eps: ', eps) 
            break

    acc = 100*test_acc / ctr


    actual = np.vstack(actual)
    adv_images = np.vstack(adv_images)
    target = np.vstack(target)
    prediction = np.vstack(prediction)
    variance = np.vstack(variance)
    corrmu = np.vstack(corrmu)
    imu = np.vstack(imu)
    corrvar = np.vstack(corrvar)
    ivar = np.vstack(ivar)
    corr_adv = np.vstack(corr_adv)
    corr_target = np.vstack(corr_target)
    i_adv = np.vstack(i_adv)
    i_target = np.vstack(i_target)

    snr = 10 * np.log10((np.sum(np.square(actual), (2, 3))) / (np.sum(np.square(actual - adv_images),(2, 3))))

    mean_snr = np.mean(snr)    
    print('snr :', mean_snr)
    print('Test Accuracy: ', acc)

    with open('../../Results/withAdv/CIFAR100_Res18/CIFAR100_Res18_AdvAttk_Results_251121_1500_{}_{}'.format(
        eps, args.adv_trgt), 'wb') as pf:
        print('saving')
        pickle.dump([actual, adv_images, target, prediction, variance, corrmu, imu, corrvar, ivar, corr_adv, i_adv, corr_target, i_target], pf)
        pf.close()

########################################################

seed_everything(41)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(41)

def main():
    
    args = parser.parse_args()

 
    args.devices = torch.device('cuda:0')
    print('Using device:', args.devices)
    
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                     std=[0.267, 0.256, 0.276])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(args.data_path, train=True, download=True, 
                                          transform=torchvision.transforms.Compose([                                             
#                                               transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                              transforms.RandomResizedCrop(32),
                                              transforms.RandomHorizontalFlip(),
                                              torchvision.transforms.ToTensor(),
                                              normalize,
                                           ])),
        batch_size=args.batch_size_train, shuffle=True, num_workers=16, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(args.data_path, train=False, download=True, 
                                          transform=torchvision.transforms.Compose([
                                              transforms.Resize(32),
                                              torchvision.transforms.ToTensor(),
                                              normalize,
                                          ])),
      batch_size=args.batch_size_train, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(args.data_path, train=False, download=True, 
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              normalize,
                                          ])),
        batch_size=args.batch_size_test, shuffle=False)


    if args.testing==False:
#         network = Net(args).to(args.devices)
        network = ResNet(18, Block, 3, args.output_size).to(args.devices)
        optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 70, 110], gamma=0.1)
#         scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0005)

        for epoch in range(1, args.epochs + 1):
            trg_loss, trg_acc = train(args, network, optimizer, train_loader, epoch)
            val_acc = validation(args, network, val_loader)
            scheduler.step()
    
        
    elif args.testing==True:
        print('Initializing Testing')
        network = Net(args)
        network.load_state_dict(torch.load(args.load_model))
        print('network loaded')
        logging.info('Model:\n{}'.format(network))
        
        if args.add_noise==True:
            print('Noise testing')
            network.eval()
            noise_std = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68]

            for noise in noise_std:
                test(args, network, test_loader, noise)

        elif args.adv_attack==True:
            print('Adv Attack testing')
            print('Target:', args.adv_trgt)
            network = network.to(args.devices)
            network.eval()
            new_model = nn.Sequential(network, SelectOutput())
            
            epsilon = [0, 0.06, 0.1, 0.3]

            c=400
            for eps in epsilon:
                if eps==0:
                    attack = torchattacks.PGD(new_model, eps=15/255, alpha=4/255, steps=50, random_start=False)
                elif eps==1:
                    attack = torchattacks.CW(new_model, c=c, kappa=0, steps=1000, lr=0.01)
                else:
                    attack = torchattacks.FGSM(new_model, eps=eps)

                test_adv(args, network, test_loader, attack, eps)
                
                
if __name__ == '__main__':
    main()
