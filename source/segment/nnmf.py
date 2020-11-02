import torch
import torch.nn as nn
import numpy as np
from utils import softminus
import math
import numbers
from torch.nn import functional as F


class SubNet(nn.ModuleList):
    def __init__(self, list):
        super(SubNet, self).__init__(list)

    def forward(self, input):
        output = input
        for l in self:
            output = l(output)
        return output


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class NNMF(nn.Module):
    def __init__(self, gmf_size, mlp_size, mlp_layers, threshold_layers):
        super(NNMF, self).__init__()
        self.gmf_size = gmf_size
        self.mlp_size = mlp_size
        self.threshold_layers = threshold_layers
        self.mlp_layers = mlp_layers
        self.embedding_activation = nn.functional.softplus
        self.mlp_activation = nn.LeakyReLU
        self.threshold_activation = nn.ReLU
        self.threshold_activation_output = nn.ReLU
        self.output_activation = nn.Sigmoid
        self.neu_mf_input_size = self.mlp_layers[-1] * (self.mlp_size > 0) + self.gmf_size
        self.mlp_input_size = 2 * self.mlp_size
        self.threshold_mlp = None
        self.mlp = None
        self.neu_mf = None
        self.num_pixels = None
        self.num_frames = None
        self.gmf_u = None
        self.gmf_v = None
        self.mlp_u = None
        self.mlp_v = None
        self.define_nn()

    def define_nn(self):
        self.threshold_mlp = SubNet([nn.Linear(1, self.threshold_layers[0]), self.threshold_activation()] +
                                    [item for t in [(nn.Linear(self.threshold_layers[j],
                                                               self.threshold_layers[j + 1]),
                                                     self.threshold_activation())
                                                    for j in range(len(self.threshold_layers) - 1)] for item in t])
        self.threshold_mlp[-1] = self.threshold_activation_output()
        self.mlp = SubNet([nn.Linear(self.mlp_input_size, self.mlp_layers[0]), self.mlp_activation()] +
                          [item for t in [(nn.Linear(self.mlp_layers[j], self.mlp_layers[j + 1]),
                                           self.mlp_activation())
                                          for j in range(len(self.mlp_layers) - 1)] for item in t])
        self.neu_mf = SubNet([nn.Linear(self.neu_mf_input_size, 1), self.output_activation()])

    def set_matrix(self, matrix2d, embedding_nmf_init=None):
        self.num_pixels = matrix2d.shape[0]
        self.num_frames = matrix2d.shape[1]

        initialize_embedding = lambda x: nn.Embedding.from_pretrained(torch.from_numpy(x).float(), freeze=False)
        get_random_init = lambda size: softminus(np.random.normal(loc=0.5, scale=0.01, size=size))

        if embedding_nmf_init:
            self.gmf_u = initialize_embedding(softminus(embedding_nmf_init[0]))
            self.gmf_v = initialize_embedding(softminus(embedding_nmf_init[1]))
        else:
            self.gmf_u = initialize_embedding(get_random_init((self.num_pixels, self.gmf_size)))
            self.gmf_v = initialize_embedding(get_random_init((self.num_frames, self.gmf_size)))

        self.mlp_u = initialize_embedding(get_random_init((self.num_pixels, self.mlp_size)))
        self.mlp_v = initialize_embedding(get_random_init((self.num_frames, self.mlp_size)))

    def init_params(self, gmf_net_init=False):
        def init_weights(m):
            if type(m) == nn.Sequential:
                try:
                    nn.init.xavier_normal_(m.weight.data, gain=1)
                    nn.init.normal_(m.bias, mean=0.0, std=0.01)
                except:
                    pass

        self.apply(init_weights)

        if gmf_net_init:
            with torch.no_grad():
                for l in self.mlp:
                    try:
                        l.weight.fill_(0.)
                        l.bias.fill_(0.)
                    except:
                        pass
                for l in self.neu_mf:
                    try:
                        l.weight.fill_(1.)
                        l.bias.fill_(0.)
                    except:
                        pass

        with torch.no_grad():
            for l in self.threshold_mlp:
                try:
                    nn.init.eye_(l.weight)
                    l.bias.fill_(0.)
                except:
                    pass

    def forward(self, pixel, frame, target):

        neu_mf_input = []
        if self.mlp_size != 0:
            mlp_input = torch.cat([self.embedding_activation(self.mlp_u(pixel)),
                                   self.embedding_activation(self.mlp_v(frame))], dim=1)
            mlp_output = self.mlp(mlp_input)
            neu_mf_input += [mlp_output]

        if self.gmf_size != 0:
            neu_mf_input += [torch.mul(self.embedding_activation(self.gmf_u(pixel)),
                                       self.embedding_activation(self.gmf_v(frame)))]

        neu_mf_input = torch.cat(neu_mf_input, dim=1)
        neu_mf_output = self.neu_mf(neu_mf_input)

        s_input = target - neu_mf_output
        s_output = self.threshold_mlp(s_input)

        return neu_mf_output, s_output

    def embedding_parameters(self):
        embedding_params = []

        if self.mlp_size != 0:
            embedding_params += list(self.mlp_u.parameters()) + list(self.mlp_v.parameters())
        if self.gmf_size != 0:
            embedding_params += list(self.gmf_u.parameters()) + list(self.gmf_v.parameters())
        return embedding_params

    def embedding_regularization(self, pixel, frame):
        loss = 0
        if self.gmf_size != 0:
            loss += torch.norm(self.embedding_activation((self.gmf_u(pixel)))) + \
                    torch.norm(self.embedding_activation((self.gmf_v(frame))))
        if self.mlp_size != 0:
            loss += torch.norm(self.embedding_activation((self.mlp_u(pixel)))) + \
                    torch.norm(self.embedding_activation((self.mlp_v(frame))))
        return loss / pixel.shape[0]

    def spatial_regularization(self, device):
        loss = 0

        def refactor_embedding(emb):
            emb_r = self.embedding_activation(emb)
            emb_r = emb_r.view([1, int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels)), -1])
            emb_r = emb_r.permute([0, 3, 1, 2])
            return emb_r

        def add_loss(embedding_weight, size):
            kernel_size = 15
            pad = list(int((kernel_size-1)/2)*np.array([1, 1, 1, 1, 0, 0, 0, 0]))
            gaussian_sm = GaussianSmoothing(channels=size, kernel_size=kernel_size, sigma=1, dim=2).to(device)
            gmf_u = refactor_embedding(embedding_weight)
            gmf_u_sq = torch.mul(gmf_u, gmf_u)
            conv_gmf = torch.nn.functional.pad(gaussian_sm(gmf_u),
                                               pad=pad, mode='constant', value=0.)
            conv_gmf_sq = torch.nn.functional.pad(gaussian_sm(gmf_u_sq), pad=pad, mode='constant', value=0.)
            return (torch.sum(gmf_u_sq.flatten()) + torch.sum(conv_gmf_sq) -
                    2 * torch.dot(gmf_u.flatten(), conv_gmf.flatten()))

        if self.gmf_size != 0: loss += add_loss(self.gmf_u.weight, self.gmf_size) / self.num_pixels
        if self.mlp_size != 0: loss += add_loss(self.mlp_u.weight, self.mlp_size) / self.num_pixels

        return loss

    def temporal_regularization(self, device):
        loss = 0

        def refactor_embedding(emb):
            emb_r = self.embedding_activation(emb)
            emb_r = emb_r.view([1, self.num_frames, -1])
            emb_r = emb_r.permute([0, 2, 1])
            return emb_r

        def add_loss(embedding_weight, size):
            kernel_size = 15
            pad = list(int((kernel_size - 1) / 2) * np.array([1, 1, 0, 0, 0, 0]))
            gaussian_sm = GaussianSmoothing(channels=size, kernel_size=kernel_size, sigma=1, dim=1).to(device)
            gmf_u = refactor_embedding(embedding_weight)
            gmf_u_sq = torch.mul(gmf_u, gmf_u)
            conv_gmf = torch.nn.functional.pad(gaussian_sm(gmf_u), pad=pad, mode='constant', value=0.)
            conv_gmf_sq = torch.nn.functional.pad(gaussian_sm(gmf_u_sq), pad=pad, mode='constant', value=0.)
            return (torch.sum(gmf_u_sq.flatten()) + torch.sum(conv_gmf_sq) -
                    2 * torch.dot(gmf_u.flatten(), conv_gmf.flatten()))

        if self.gmf_size != 0: loss += add_loss(self.gmf_v.weight, self.gmf_size) / self.num_frames
        if self.mlp_size != 0: loss += add_loss(self.mlp_v.weight, self.mlp_size) / self.num_frames

        return loss
