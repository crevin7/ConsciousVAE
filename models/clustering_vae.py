from importlib.util import module_for_loader
from tokenize import String
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from sklearn.cluster import KMeans

#these functions here should be moved in a different file
def conv_module(in_channels, out_channels):
    module = nn.Sequential(
        nn.Conv2d(in_channels, out_channels=out_channels,
                kernel_size= 4, stride= 2, padding  = 1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU())
    return module

def linear_module(in_channels, out_channels):
    module = nn.Sequential(
        nn.Linear(in_channels, out_features=out_channels),
        nn.LeakyReLU())
    return module

def conv_transpose_module(in_channels, out_channels):
    module = nn.Sequential(
        nn.ConvTranspose2d(in_channels,
                        out_channels,
                        kernel_size=4,
                        stride = 2,
                        padding=1,
                        output_padding=0),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU())
    return module

class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 kind: String = 'conv',
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.encoder_dims = kwargs.get("encoder_dims", hidden_dims)
        self.decoder_dims = kwargs.get("decoder_dims", hidden_dims[::-1])
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.kind = kind

        self._build_encoder()
        self._build_decoder()

        
    def _build_encoder(self)->None:
        modules = []
        in_channels = self.in_channels

        if self.kind == 'conv':
            module = conv_module
            # this quantity here should be defined by the user, as it depends by the kernel 
            # and the depth of the network
            n_units = 4 
        elif self.kind == 'linear':
            module = linear_module
            n_units = 1
        else:
            ValueError(f"kind {self.kind} not implemented")

        for h_dim in self.encoder_dims:
            modules.append(
                module(in_channels, h_dim)
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(in_channels*n_units, self.latent_dim)
        self.fc_var = nn.Linear(in_channels*n_units, self.latent_dim)


    def _build_decoder(self)->None:

        modules = []

        if self.kind == 'conv':
            module = conv_transpose_module
            # this quantity here should be defined by the user, as it depends by the kernel 
            # and the depth of the network
            self.n_units = 4*2 
        elif self.kind == 'linear':
            module = linear_module
            self.n_units = 1
        else:
            ValueError(f"kind {self.kind} not implemented")

        self.decoder_input = nn.Linear(self.latent_dim,
         self.decoder_dims[0] * (self.n_units)*(self.n_units))


        for i in range(len(self.decoder_dims) - 1):
            modules.append(
                module(self.decoder_dims[i],
                        self.decoder_dims[i + 1])
            )
            
        self.decoder = nn.Sequential(*modules)

        if self.kind == 'conv':
            self.final_layer = nn.Sequential(
                                nn.Conv2d(self.decoder_dims[-1],
                                out_channels= self.in_channels,
                                kernel_size=4, stride=2, padding=1),
                                nn.Tanh())
        else:
            self.final_layer = nn.Sequential(
                                nn.Linear(self.decoder_dims[-1],
                                out_features= self.in_channels),
                                nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        if self.kind == 'conv':
          result = result.view(-1, self.decoder_dims[0], self.n_units,
           self.n_units)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def assign_cluster(self, z: Tensor, clusters: Tensor):
        assert(z.size()[1]) == clusters.size()[1] == self.laltent_dim
        dist = torch.norm(clusters - z, dim=1)
        closest_cluster = dist.topk(1, largest=False)
        return closest_cluster

    def forward(self, input: Tensor, clusters: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        clust_mu = self.assign_cluster(z, clusters=clusters)
        return  [self.decode(z), input, self.decode(clust_mu), mu, log_var, clust_mu]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        clust_recons = args[2]
        mu = args[3]
        log_var = args[4]
        clust_mu = args[5]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        mean_rec_weight = kwargs['mean_rec_weight'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - (mu - clust_mu)** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + mean_rec_weight*clust_recons + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]