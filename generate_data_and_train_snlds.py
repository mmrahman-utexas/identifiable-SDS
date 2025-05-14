import argparse
import os

import sys
import time

import cv2
import numpy as np
from scipy.special import logsumexp, comb
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from dataloaders.BouncingBallDataLoader import MdDataLoader
from models.modules import MLP
from models.VariationalSNLDS import VariationalSNLDS



import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from utils.transitions import (get_trans_mat, func_cosine_with_sparsity, func_polynomial, 
    func_leaky_relu, func_softplus_with_sparsity, sample_adj_mat)

def save_checkpoint(state, filename='model'):
    os.makedirs("results/models_sds/", exist_ok=True)
    torch.save(state, "results/models_sds/" + filename + '.ckpt')

def parse_args():

    parser = argparse.ArgumentParser(description='Ar-HMM Data Gen and train')
    parser.add_argument('--seeds', default=[24], nargs="+", type=int, metavar='N', help='number of seeds (multiple seeds run multiple experiments)')
    parser.add_argument('--dim_obs', default=2, type=int, metavar='N', help='number of dimensions')
    parser.add_argument('--dim_latent', default=2, type=int, metavar='N', help='number of latent dimensions')
    parser.add_argument('--num_states', default=4, type=int, metavar='N', help='number of states')
    parser.add_argument('--sparsity_prob', default=0.0, type=float, metavar='N', help='sparsity probability')
    parser.add_argument('--data_type', default='cosine', type=str, help='Type of data generated (cosine|poly)')
    parser.add_argument('--generate_data', action='store_true', help='Generate data and then train')
    parser.add_argument('--no-train', action='store_true', help='Activate no train')
    parser.add_argument('--images', action='store_true', help='Use images')
    parser.add_argument('--device', default='0', type=str, help='Device to use')
    parser.add_argument('--degree', default=3, type=int, metavar='N', help='degree of polynomial')
    parser.add_argument('--restarts_num', default=10, type=int, metavar='N', help='number of random restarts')
    parser.add_argument('--seq_length', default=200, type=int, metavar='N', help='number of timesteps')
    parser.add_argument('--num_samples', default=5000, type=int, metavar='N', help='number of samples')
    
    return parser.parse_args()


def train(path, num_states, dim_obs, dim_latent, T, data_size, sparsity_prob, data_type, device, seed):

    if args.images:
        dl = MdDataLoader(path)
        exp_name = 'inferred_params_images_N_{}_T_{}_dim_latent_{}_state_{}_sparsity_{}_net_{}_seed_{}'.format(data_size,
                T, dim_latent, num_states, sparsity_prob, data_type, seed)
    else:
        dl = TensorDataset(torch.from_numpy(np.load(path)))
        exp_name = 'inferred_params_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}'.format(data_size,
                T, dim_latent, dim_obs, num_states, sparsity_prob, data_type, seed)
    final_temperature = 1

    ## Hyperparameters (highly advisable to modify on data type and size)
    ## In this configuration all the models use the temperature scheduling indicated in Appendix F.3
    ## https://arxiv.org/abs/2305.15925
    ## If the models runs into state collapse, we advise increasing the temperature and decay rate.
    ## If the temperature is increased, it is advised to increase the scheduler_epochs to prevent fake convergence of Q
    if args.images:
        pre_train_check = 10
        init_temperature = 10
        iter_update_temp = 50
        iter_check_temp = 200
        epoch_num = 200
        learning_rate = 5e-4
        gamma_decay = 0.8
        scheduler_epochs = 80
        decay_rate = 0.975
    else:           
        pre_train_check = 5
        init_temperature = 5
        iter_update_temp = 50
        iter_check_temp = 1000
        epoch_num = 100
        learning_rate = 5e-4
        gamma_decay = 0.5
        scheduler_epochs = 40
        decay_rate = 0.9
    
    for restart_num in range(args.restarts_num):
        best_elbo = -torch.inf
        if args.images:
            dataloader = DataLoader(dl, batch_size=6, shuffle=True)
        else:
            dataloader = DataLoader(dl, batch_size=50, shuffle=True)
        model = VariationalSNLDS(dim_obs, dim_latent, 64, num_states, encoder_type='video' if images else 'recurent', device=device, annealing=False, inference='alpha', beta=0)
        # Useful for setting a smaller transition network to avoid overfitting
        model.transitions = torch.nn.ModuleList([MLP(dim_latent, dim_latent, 16, 'cos') for _ in range(num_states)]).to(device).float()

        model.temperature = init_temperature
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_epochs, gamma=gamma_decay)
        iterations = 0
        model.beta = 0
        mse = 1e5
        model.Q.requires_grad_(False)
        model.pi.requires_grad_(False)
        for epoch in range(0, epoch_num):
            if epoch >= pre_train_check and mse > 6e3:
                print("MSE too high, stopping training")
                break
            if epoch >= pre_train_check and epoch < scheduler_epochs//4:
                model.beta = 1
                if args.images: # With images we will use high temperature annealing. No need for long warmups
                    model.Q.requires_grad_(True)
                    model.pi.requires_grad_(True)
            elif epoch >= scheduler_epochs//4:
                model.Q.requires_grad_(True)
                model.pi.requires_grad_(True)
            end = time.time()
            for i, (sample,) in enumerate(dataloader, 1):
                if args.images:
                    B, T, C, H, W = sample.size()
                else:
                    B, T, D = sample.size()
                obs_var = Variable(sample[:,:].float(), requires_grad=True).to(device)
                optimizer.zero_grad()
                x_hat, _, _, losses = model(obs_var)
                # Compute loss and optimize params
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if args.images:
                    mse = torch.nn.functional.mse_loss(x_hat.reshape(B, T, C, H, W), obs_var, reduction='sum')/(B)
                else:
                    mse = torch.nn.functional.mse_loss(x_hat, obs_var, reduction='sum')/(B)
                batch_time = time.time() - end
                end = time.time()   
                iterations +=1
                if iterations%iter_update_temp==0 and iterations >= iter_check_temp:
                    model.temperature = model.temperature*decay_rate
                    model.temperature = max(model.temperature, final_temperature)
                if i%5==0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time:.3f}\t'
                        'ELBO {loss:.4e}\t MSE: {mse:.4e}\t MSM: {msm:.4e}'.format(
                        epoch, i, len(dataloader), batch_time=batch_time, 
                        loss=losses['elbo'], mse=mse, msm=losses['msm_loss']))
                    sys.stdout.flush()

            if epoch%2==0:
                print((model.Q/model.temperature).softmax(-1))
                print((model.pi/model.temperature).softmax(-1))
                print(model.temperature)
            sys.stdout.flush()
            scheduler.step()
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict()
            }, filename=exp_name+ '_restart_{:02d}'.format(restart_num))
            if best_elbo < losses['elbo']:
                best_elbo = losses['elbo']
                save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict()
                }, filename=exp_name+'_best_model')

if __name__=="__main__":
    
    args = parse_args()
    data_type = args.data_type
    data_size = args.num_samples
    T = args.seq_length
    dim_obs = args.dim_obs
    dim_latent = args.dim_latent
    num_states = args.num_states
    sparsity_prob = args.sparsity_prob
    images = args.images
    device = torch.device(args.device)
    degree = args.degree
    restarts_num = args.restarts_num
    for k, seed in enumerate(args.seeds):
        # if args.generate_data:
        #     generate_data(seed, num_states, dim_obs, dim_latent, T, data_size, sparsity_prob, data_type, degree, save=False, images=images)
        if not images:
            path = 'data/latent_variables/obs_train_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}.npy'.format(data_size,T, dim_latent, dim_obs, num_states, sparsity_prob, data_type, seed)
        else:
            # path = "data/latent_variables/images_train_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}/".format(data_size,T, dim_latent, dim_obs, num_states, sparsity_prob, data_type, seed)
            path = "data/M_D/data_train.pt"
        if not args.no_train:
            train(path, num_states, dim_obs, dim_latent, T, data_size, sparsity_prob, data_type, device, seed)
            print("Trained Seed:", seed)
        sys.stdout.flush()
