import torch
import abc


class Sampler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_points(self,pc_input):
        pass


class NormalPerPoint(Sampler):

    def __init__(self, global_sigma=1.0, local_sigma=0.01, dimension=3):
        self.dimension=3
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points_global(self, n_points, batch_size=None, device='cpu'):
        if batch_size:
            return (torch.rand(batch_size, n_points, self.dimension, device=device) * (self.global_sigma * 2)) - self.global_sigma
        else:
            return (torch.rand(n_points, self.dimension, device=device) * (self.global_sigma * 2)) - self.global_sigma

    def get_points_local(self, pc_input, n_points):
        if len(pc_input.size()) == 3:
            return pc_input[:,:n_points, :] + (torch.randn_like(pc_input[:,:n_points, :]) * self.local_sigma)
        elif len(pc_input.size()) == 2:
            return pc_input[:n_points, :] + (torch.randn_like(pc_input[:n_points, :]) * self.local_sigma)


    def get_points(self, pc_input, n_points=None, local_ratio=0.8):
        if len(pc_input.size()) == 3:
            batch_size, sample_size, dim = pc_input.shape
        elif len(pc_input.size()) == 2:
            sample_size, dim = pc_input.shape
            batch_size = None

        if not n_points: n_points = sample_size

        n_points_local = int(local_ratio * n_points)
        n_points_global = n_points - n_points_local

        return torch.cat([
            self.get_points_global(n_points_global, batch_size, device=pc_input.device),
            self.get_points_local(pc_input, n_points_local)
            ], dim=1)
