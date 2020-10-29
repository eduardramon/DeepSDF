import torch
import abc


class Sampler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_points(self,pc_input):
        pass


class NormalPerPoint(Sampler):

    def __init__(self, global_sigma, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input, local_ratio=0.8, local_sigma=None):
        batch_size, sample_size, dim = pc_input.shape

        samples_size_local = int(local_ratio*sample_size)
        samples_size_global = sample_size - samples_size_local
        if local_sigma is not None:
            sample_local = pc_input[:,:samples_size_local, :] + (torch.randn_like(pc_input[:,:samples_size_local, :]) * local_sigma.unsqueeze(-1))
        else:
            sample_local = pc_input[:,:samples_size_local, :] + (torch.randn_like(pc_input[:,:samples_size_local, :]) * self.local_sigma)

        sample_global = (torch.rand(batch_size, samples_size_global, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample