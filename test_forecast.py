import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from dataloaders.BouncingBallDataLoader import MdDataLoader
from models.VariationalSNLDS import VariationalSNLDS
from models.modules import MLP
from metrics import reconstruction_mse_per_frame, mean_corr_coef
from plotting import show_images


def manual_forecast(model, prefix, forecast_len):
    device = prefix.device
    B, T, C, H, W = prefix.shape

    with torch.no_grad():
        z_sample, _, _ = model._encode_obs(prefix)
    z_sample = z_sample.view(B, T, -1)

    log_evidence = model._compute_local_evidence(z_sample)
    gamma, _    = model._compute_posteriors(log_evidence)
    last_disc   = gamma[:, -1, :].argmax(-1)
    last_cont   = z_sample[:, -1, :].clone()

    latent_pred = torch.zeros(B, forecast_len, model.latent_dim, device=device)
    Q = model.Q

    for t in range(forecast_len):
        logits    = Q[last_disc]                         # (B, num_states)
        last_disc = Categorical(logits=logits).sample()  # (B,)
        next_cont = []
        for b in range(B):
            z_t = last_cont[b : b+1]
            z_t = model.transitions[last_disc[b]](z_t)
            next_cont.append(z_t)
        last_cont          = torch.cat(next_cont, dim=0)
        latent_pred[:, t]  = last_cont

    x_hat = model._decode(latent_pred.view(B*forecast_len, -1))
    x_hat = x_hat.view(B, forecast_len, C, H, W)
    return x_hat, latent_pred


def main():
    # --- USER PARAMS ---
    ckpt_path    = 'results/models_sds/inferred_params_images_N_249_T_4000_dim_latent_2_state_4_sparsity_0.0_net_cosine_seed_1111_best_model.ckpt'
    test_data    = 'data/M_D/data_test.pt'
    batch_size   = 8
    forecast_len = 30
    # -------------------

    # derive an output dir from the checkpoint filename
    model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    base_dir   = os.path.dirname(ckpt_path)
    output_dir = os.path.join(base_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # load full test file (for z_gt and imgs)
    data_dict = torch.load(test_data)
    z_gt_all   = data_dict['xt']          # (N, T, latent_dim)
    # img_all    = data_dict['yt']          # (N, T, H, W)

    # prepare DataLoader
    test_ds = MdDataLoader(test_data)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # device & model
    device     = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    ckpt       = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model']

    latent_dim = z_gt_all.shape[2]
    num_states = state_dict['Q'].shape[0]
    model = VariationalSNLDS(
        obs_dim=2, latent_dim=latent_dim, hidden_dim=64,
        num_states=num_states, encoder_type='video', device=device
    )
    model.transitions = torch.nn.ModuleList([MLP(latent_dim, latent_dim, 16, 'cos') for _ in range(num_states)]).to(device).float()
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # figure out prefix length
    T_full     = z_gt_all.shape[1]
    prefix_len = T_full - forecast_len

    # accumulators
    all_pred_imgs = []
    all_gt_imgs   = []
    all_pred_z    = []
    all_gt_z      = []

    for batch_idx, (images,) in enumerate(test_ld):
        images = images.to(device)
        B, _, C, H, W = images.shape

        start = batch_idx * batch_size
        end   = start + B

        # gt_imgs = img_all[start:end, prefix_len:]           # (B, F, H, W)
        # gt_imgs = gt_imgs[:, :, None, :, :]                  # → (B, F, 1, H, W)
        gt_imgs = images[:, prefix_len:]           # (B, F, C, H, W)
        gt_z    = z_gt_all[start:end, prefix_len:]           # (B, F, latent_dim)

        with torch.no_grad():
            pred_imgs, pred_z = manual_forecast(model, images[:, :prefix_len], forecast_len)

        all_pred_imgs.append(pred_imgs.cpu().numpy())
        all_gt_imgs.  append(gt_imgs.cpu().numpy())
        all_pred_z.   append(pred_z.cpu().numpy())
        all_gt_z.     append(gt_z)

    # stack everything
    pred_imgs = np.concatenate(all_pred_imgs, axis=0)
    gt_imgs   = np.concatenate(all_gt_imgs,   axis=0)
    pred_z    = np.concatenate(all_pred_z,     axis=0)
    gt_z      = np.concatenate(all_gt_z,       axis=0)

    # compute metrics only once
    mse_mean, mse_std = reconstruction_mse_per_frame(pred_imgs, gt_imgs)
    corr_score, corr_cc     = mean_corr_coef(
        gt_z.reshape(-1, latent_dim),
        pred_z.reshape(-1, latent_dim)
    )

    # save metrics to JSON
    metrics = {
        'mse_mean':        float(mse_mean),
        'mse_std':         float(mse_std),
        'mean_corr_coef':  float(corr_score),
        'mean_corr_coef_cc': corr_cc.tolist(),
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'Saved metrics → {os.path.join(output_dir, "metrics.json")}')

    # visualize the very first sample
    vis_path = os.path.join(output_dir, 'forecast_sample0.png')
    show_images(gt_imgs[:10], pred_imgs[:10], vis_path)
    print(f'Saved visualization → {vis_path}')


if __name__ == '__main__':
    main()