import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import m_elbo
from vae_vision import Encoder_vision, Decoder_vision
from vae_audio import Encoder_audio, Decoder_audio

eta = 1e-6
eps = 1e-8

class MultiVAE(nn.Module):
    def __init__(self, latent_dim, device):
        super(MultiVAE, self).__init__()
        self.encoder_vision = Encoder_vision(latent_dim)
        self.decoder_vision = Decoder_vision(latent_dim)
        self.encoder_audio = Encoder_audio(latent_dim)
        self.decoder_audio = Decoder_audio(latent_dim)
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, vision=None, audio=None):
        mu, logvar, mu_vision, logvar_vision, mu_audio, logvar_audio = \
            self.encoder(vision=vision, audio=audio)
        if vision is not None:
            latent_vision = self.reparameterize(mu_vision, logvar_vision)
            vision_recon = self.decoder_vision(latent_vision)
        else:
            vision_recon = None
        if audio is not None:
            latent_audio = self.reparameterize(mu_audio, logvar_audio)
            audio_recon = self.decoder_audio(latent_audio)
        else:
            audio_recon = None
        return vision_recon, audio_recon, mu, logvar

    def encoder(self, vision=None, audio=None):
        if vision is not None:
            mu_vision, logvar_vision = self.encoder_vision(vision)
        else:
            mu_vision, logvar_vision = None, None
        if audio is not None:
            mu_audio, logvar_audio = self.encoder_audio(audio)
        else:
            mu_audio, logvar_audio = None, None

        if vision is None:
            mu, logvar = mu_audio, logvar_audio
        elif audio is None:
            mu, logvar = mu_vision, logvar_vision
        else:
            mu, logvar = mixture_2experts(self.latent_dim, mu_vision, logvar_vision, mu_audio, logvar_audio)
        return mu, logvar, mu_vision, logvar_vision, mu_audio, logvar_audio
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps_ = Variable(std.data.new(std.size()).normal_())
            return eps_.mul(std).add_(mu)
        else:
            return mu
        
    # ... (残りの関数は変更なし) ...



def mixture_2experts(dim, mu1, logvar1, mu2, logvar2):
    mu = torch.cat((mu1[:, 0:dim // 2], mu2[:, dim // 2:dim]), dim=1)
    logvar = torch.cat((logvar1[:, 0:dim // 2], logvar2[:, dim // 2:dim]), dim=1)
    return mu, logvar
# train関数も同様に触覚関連のコードを削除します。
# ... (train関数の修正版) ...
def train(args, model, data_loader, optimizer, device='cuda', mu_prior=None, var_prior=None, save_epoch=0, desc=None):
    model.train()
    loss_average = 0
    loss_vision = 0
    loss_audio = 0
    means = torch.zeros(1, args.latent_dim)
    logvars = torch.zeros(1, args.latent_dim)
    for batch_idx, data in enumerate(data_loader):
        if desc == 'vision_audio':
            vision = data[0].to(device)
            audio = data[1].to(device)
        else:
            vision = data[0].to(device)
            audio = data[1].to(device)

        vision_recon, audio_recon, mu, logvar = model(vision=vision, audio=audio)
        loss_vision = 0
        loss_audio = 0

        if mu_prior is not None:
            batch_head = batch_idx * args.batch_size
            batch_end = batch_head + args.batch_size
            mu_prior_batch = mu_prior[batch_head:batch_end]
            var_prior_batch = var_prior[batch_head:batch_end]
            loss, la, lb = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                          recon_audio=audio_recon, audio=audio,
                          mu=mu, logvar=logvar, mu_prior=mu_prior_batch, var_prior=var_prior_batch,
                          dim=args.latent_dim, variational_beta=args.variational_beta)
        else:
            loss, la, lb = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                          recon_audio=audio_recon, audio=audio,
                          mu=mu, logvar=logvar, variational_beta=args.variational_beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_average += loss.item()
        loss_vision += la
        loss_audio += lb
        if save_epoch != 0:
            for mean in mu:
                mean = mean.cpu()
                means = torch.cat((means, mean.unsqueeze(0)), dim=0)
            for lv in logvar:
                lv = lv.cpu()
                logvars = torch.cat((logvars, lv.unsqueeze(0)), dim=0)
    loss_average /= len(data_loader.dataset)
    return loss_average, means[1:], logvars[1:], loss_vision, loss_audio
