import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import grad

class default_weight_args:
    def __init__(self):
        super(default_weight_args, self).__init__()
        self.fc_input_mean_mu = 0
        self.fc_input_mean_sigma = 0.05
        self.fc_input_mean_bias = 0.0001
        self.fc_input_sigma_min = -12
        self.fc_input_sigma_max = -1.0
        self.fc_input_sigma_bias = 0.0001

        self.conv_input_mean_mu = 0
        self.conv_input_mean_sigma = 0.1
        self.conv_input_mean_bias = 0.0001
        self.conv_input_sigma_min = -12
        self.conv_input_sigma_max = -2.2
        self.conv_input_sigma_bias = 0.0001


global_layer_default = default_weight_args()

class VDP_Flatten(nn.Module):
    def __init__(self):
        super(VDP_Flatten, self).__init__()

    def forward(self, mu, sigma):
        mu_flat = torch.flatten(mu, start_dim=1)

        sigma_flat = torch.zeros(mu_flat.shape[0], mu_flat.shape[1], mu_flat.shape[1], device=sigma.device)

        batch = sigma.shape[0]
        for i in range(batch):
            sigma_flat[i, :] = torch.block_diag(*sigma[i, :])

        return mu_flat, sigma_flat


class VDP_Conv2D(nn.Module):
    
    def __init__(self, in_channels, out_channels,
                 kernel_size=(5, 5), stride=1, padding=0, dilation=1,
                 groups=1, bias=False, padding_mode='zeros', weight_args=global_layer_default,
                 input_flag=False):

        super(VDP_Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_flag = input_flag
        self.stride = stride

        self.bias = bias
        self.padding = padding

        # Standard Conv Layer
        self.mean = nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding, dilation,
                                   groups, bias, padding_mode)
        # Conv Layer Initializing
        nn.init.normal_(self.mean.weight, mean=weight_args.conv_input_mean_mu, std=weight_args.conv_input_mean_sigma)
        
        # Sigma Conv Layer
        self.sigma_weight = nn.Parameter(torch.zeros([out_channels]), requires_grad=True)
        # Sigma Conv Layer Initializing
        nn.init.uniform_(self.sigma_weight, weight_args.conv_input_sigma_max)
        
        if self.bias:
            self.mean.bias.data.fill_(weight_args.conv_input_mean_bias)
            self.sigma_bias = nn.Parameter(torch.tensor(weight_args.conv_input_sigma_bias), requires_grad=True)

    def forward(self, mu, sigma=0):
        # First layer recieving the image input
        mu_z = self.mean(mu)

        batch_size = mu.shape[0]
        num_channels = mu.shape[1]
        img_size = mu.shape[-1]
        if self.padding!=0:
            mu = F.pad(mu, (1,1,1,1))
        mu_patches = mu.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        mu_patches = mu_patches.permute(0, 2, 3, 1, 4, 5).contiguous()        
        mu_patches = mu_patches.view(*mu_patches.size()[:3], -1)
        mu_matrix = torch.reshape(mu_patches, (batch_size, -1, self.kernel_size*self.kernel_size*num_channels))
        mu_muTranspose = torch.matmul(mu_matrix, mu_matrix.permute(0, 2, 1))
        mu_muTranspose = torch.ones([1, 1, 1, self.out_channels], device=mu.device) * torch.unsqueeze(mu_muTranspose, axis=-1)
        
        if self.bias:
            sigma_z = torch.mul(torch.log1p(torch.exp(self.sigma_weight)), mu_muTranspose).permute(0, 3, 1, 2) + self.sigma_bias
        else:
            sigma_z = torch.mul(torch.log1p(torch.exp(self.sigma_weight)), mu_muTranspose).permute(0, 3, 1, 2)

        if not self.input_flag:
            # Subsequent layer, not the first one
            sigma_in = sigma
            mu_in = mu
            diag_sigma = torch.diagonal(sigma, dim1=-2, dim2=-1)
            diag_sigma = diag_sigma.reshape(batch_size, num_channels, img_size, img_size)
            if self.padding!=0:
                diag_sigma = F.pad(diag_sigma, (1,1,1,1))
            sig_patches = diag_sigma.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
            sig_patches = sig_patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            sig_patches = sig_patches.view(*sig_patches.size()[:3], -1)
            diag_sigma_g = torch.reshape(sig_patches,[batch_size, -1, self.kernel_size * self.kernel_size * num_channels])            
            mu_cov_square = torch.reshape(torch.mul(self.mean.weight, self.mean.weight), 
                                          [self.kernel_size * self.kernel_size * num_channels, self.out_channels])
            mu_wT_sigmags_mu_w1 = torch.matmul(diag_sigma_g, mu_cov_square)
            mu_wT_sigmags_mu_w = torch.diag_embed(mu_wT_sigmags_mu_w1.permute(0, 2, 1), dim1=-2, dim2=-1)
                        
            trace = torch.sum(diag_sigma_g, 2, keepdim=True)
            trace = torch.ones([1, 1, self.out_channels], device=mu.device) * trace
            trace = torch.mul(torch.log1p(torch.exp(self.sigma_weight)), trace).permute(0, 2, 1)
            trace1 = torch.diag_embed(trace)

            sigma_z = trace1 + mu_wT_sigmags_mu_w + sigma_z
        return mu_z, sigma_z

    def kl_loss_term(self):
        #KL Regularization term added to the loss.
        
        c_s = torch.log1p(torch.exp(self.sigma_weight))

        kl_loss = -0.5 * torch.mean((self.kernel_size * self.kernel_size) * torch.log(c_s) +
                                    (self.kernel_size * self.kernel_size) -
                                    torch.norm(self.mean.weight) ** 2 -
                                    (self.kernel_size * self.kernel_size) * c_s)
        return kl_loss


class VDP_Relu(nn.Module):

    def __init__(self, inplace=False):

        super(VDP_Relu, self).__init__()
        self.relu = nn.ReLU(inplace)

    def forward(self, mu, sigma):
   
        mu_g = self.relu(mu)

        activation_gradient = grad(mu_g.sum(), mu, retain_graph=True)[0]  

        if len(mu_g.shape) == 2:

            grad_square = torch.bmm(activation_gradient.unsqueeze(2), activation_gradient.unsqueeze(1))

            sigma_g = torch.mul(sigma, grad_square)
        else:
            gradient_matrix = activation_gradient.permute([0, 2, 3, 1]).view(activation_gradient.shape[0], -1, mu_g.shape[1]).unsqueeze(3)

            grad1 = gradient_matrix.permute([0, 2, 1, 3])
            grad2 = grad1.permute([0, 1, 3, 2])

            grad_square = torch.matmul(grad1, grad2)

            sigma_g = torch.mul(sigma, grad_square)  
        return mu_g, sigma_g


class VDP_Maxpool(nn.Module):

    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True, ceil_mode=False, padding_mode='zeros'):
 
        super(VDP_Maxpool, self).__init__()
        self.maxPooling = nn.MaxPool2d(kernel_size, kernel_size, padding, dilation, return_indices, ceil_mode)

    def forward(self, mu, sigma):
  
        mu_p, argmax1 = self.maxPooling(mu)

        argmax = argmax1.reshape(argmax1.shape[0], argmax1.shape[1], argmax1.shape[2] ** 2)

        argmax_indices = argmax.repeat(1, 1, argmax.shape[-1]).reshape(argmax.shape[0], argmax.shape[1], argmax.shape[-1],
                                                                       argmax.shape[-1])

        index_fix = argmax.unsqueeze(3) * sigma.shape[-1]
        sigma_indexes = argmax_indices + index_fix

        sigma_p = torch.gather(sigma.reshape(argmax.shape[0], argmax.shape[1], -1), dim=2,
                               index=sigma_indexes.reshape(
                                   argmax.shape[0], argmax.shape[1], -1)).reshape(argmax.shape[0], argmax.shape[1], 
                                                                                  argmax.shape[2], argmax.shape[2])
        return mu_p, sigma_p


class VDP_FullyConnected(nn.Module):

    def __init__(self, in_features, out_features, bias=False, weight_args=global_layer_default,
                 input_flag=False):

        super(VDP_FullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.input_flag = input_flag

        self.mean = nn.Linear(in_features, out_features, bias)
        nn.init.normal_(self.mean.weight, mean=weight_args.fc_input_mean_mu, std=weight_args.fc_input_mean_sigma)

        self.sigma_weight = nn.Parameter(torch.zeros([1, out_features]), requires_grad=True)
        nn.init.uniform_(self.sigma_weight, a=weight_args.fc_input_sigma_min, b=weight_args.fc_input_sigma_max)
        
        if self.bias:
            self.mean.bias.data.fill_(weight_args.fc_input_mean_bias)
            self.sigma_bias = nn.Parameter(torch.tensor(weight_args.fc_input_sigma_bias), requires_grad=True)

    def forward(self, mu, sigma=0):

        if self.input_flag:
            # First layer recieving the image input
            mu_f = self.mean(mu)

            mu_pT_mu_p = torch.matmul(mu.unsqueeze(1), mu.unsqueeze(2)).squeeze(1)
            if self.bias:
                sigma_1 = torch.log1p(torch.exp(self.sigma_weight)).repeat(mu.shape[0], 1) * mu_pT_mu_p.repeat(1, self.out_features)
                + self.sigma_bias
            else:
                sigma_1 = torch.log1p(torch.exp(self.sigma_weight)).repeat(mu.shape[0], 1) * mu_pT_mu_p.repeat(1, self.out_features)
            sigma_out = torch.diag_embed(sigma_1, dim1=1)

        else:
            # Subsequent layer after the first one
            # Mu Weight * Mu in
            mu_f = self.mean(mu)
            
            # Mean Weight^2 * sigma in
            muh_w = self.mean.weight.unsqueeze(0)
            muhT_sigmab_mu = torch.matmul(torch.matmul(muh_w, sigma), muh_w.permute([0, 2, 1]))
            tr_diag_sigma = torch.diagonal(sigma, dim1=1, dim2=2).sum(1).unsqueeze(1)
            mu_pT_mu_p = torch.matmul(mu.unsqueeze(1), mu.unsqueeze(2)).squeeze(1)

            # tr(Sigma Weights * Sigma In)
            # Mu in ^2 * Sigma Weights
            if self.bias:
                sigma_weight_out = torch.log1p(torch.exp(self.sigma_weight)).repeat(mu.shape[0], 1) * (tr_diag_sigma.repeat(
                    1, self.out_features) + mu_pT_mu_p.repeat(1, self.out_features)) + self.sigma_bias
            else:
                sigma_weight_out = torch.log1p(torch.exp(self.sigma_weight)).repeat(mu.shape[0], 1) * (tr_diag_sigma.repeat(
                    1, self.out_features) + mu_pT_mu_p.repeat(1, self.out_features))

            diag_sigma_weight_out = torch.diag_embed(sigma_weight_out, dim1=1)

            # tr(Sigma Weights * Sigma In) + Mean Weight^2 * sigma in + Mu in ^2 * Sigma Weights
            sigma_out = muhT_sigmab_mu + diag_sigma_weight_out

        return mu_f, sigma_out

    def kl_loss_term(self):

        f_s = torch.log1p(torch.exp(self.sigma_weight))
        kl_loss = -0.5 * torch.mean((self.in_features * torch.log(f_s)) +
                                      self.in_features -
                                      torch.norm(self.mean.weight)**2 -
                                      (self.in_features * f_s))
        return kl_loss


class VDP_Softmax(nn.Module):
 
    def __init__(self, dim=1):

        super(VDP_Softmax, self).__init__()
        self.softmax = nn.Softmax(dim)

    def forward(self, mu, sigma):
 
        mu_y = self.softmax(mu)

        grad_f1 = torch.bmm(mu_y.unsqueeze(2), mu_y.unsqueeze(1))
        diag_f = torch.diag_embed(mu_y, dim1=1)
        grad_soft = diag_f - grad_f1
        sigma_y = torch.matmul(grad_soft, torch.matmul(sigma, grad_soft.permute(0, 2, 1)))
        
        mask_ = torch.eye(sigma_y.shape[-1], device=sigma_y.device)
        mask_ = mask_.reshape((1, sigma_y.shape[-1], sigma_y.shape[-1]))
        mask = mask_.repeat(sigma_y.shape[0],1,1) 
        sigma_diag = torch.diag_embed(torch.abs(torch.diagonal(sigma_y, dim1=-2, dim2=-1)))
        sigma_y_ = sigma_diag+(1.-mask)*sigma_y
        
        return mu_y, sigma_y_

    
class VDP_BatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True):
  
        super(VDP_BatchNorm, self).__init__()
        self.eps = eps
        self.batchNorm = torch.nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, mu, sigma):
  
        mu_bn = self.batchNorm(mu)
        
        var = torch.var(mu,(0))
        sigma_bn = (torch.mul(sigma, 1 / (var + self.eps)))

        return mu_bn, sigma_bn
    

class VDP_BatchNorm2D(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True):
     
        super(VDP_BatchNorm2D, self).__init__()
        self.eps = eps
        self.batchNorm = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, mu, sigma):
      
        mu_bn = self.batchNorm(mu)
        
        var = torch.var(mu,(0,2,3))
        sigma_bn = (torch.mul(sigma.permute(0,2,3,1), 1 / (var + self.eps))).permute(0,3,1,2)

        return mu_bn, sigma_bn    

