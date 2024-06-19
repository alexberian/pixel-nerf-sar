import sys
from torch import nn
import torch

#  import torch_scatter
import torch.autograd.profiler as profiler


def get_combine_module(combine_type):
    """
    return a view combiner module based on the combine_type
    """

    combine_str_to_module = {
        "average": "VanillaPixelnerfViewCombiner",
        "max": "VanillaPixelnerfViewCombiner",
        "error.5": "CamDistanceAngleErrorCombiner",
    }

    if combine_type in combine_str_to_module:
        return globals()[combine_str_to_module[combine_type]]()
    else:
        raise NotImplementedError("Unsupported combine type " + combine_type)


def calculate_relative_pose(src_poses, target_poses):
    """
    Calculate the relative pose between the source and target poses
    to go from source view to target view it's
    P_s_to_t = [ [ R_t^T R_s , R_t^T (t_s - t_t) ],
                 [ 0         , 1                 ] ]
    :param src_poses (SB, NS, 4, 4) source poses
    :param target_poses (SB, ray_batch_size, 4, 4) target poses
    :return (SB, NS, ray_batch_size, 4, 4) relative poses
    """
    # Reshape for broadcasting
    SB, NS, _, _ = src_poses.shape
    _, ray_batch_size, _, _ = target_poses.shape
    src_poses = src_poses.reshape(SB, NS, 1, 1, 4, 4)
    target_poses = target_poses.reshape(SB, 1, ray_batch_size, 1, 4, 4)

    # Extract rotation and translation
    R_s = src_poses[..., :3, :3]     # (SB, NS, 3, 3)
    R_t = target_poses[..., :3, :3]  # (SB, 1, ray_batch_size, 3, 3)
    t_s = src_poses[..., :3, 3:4]    # (SB, NS, 3, 1)
    t_t = target_poses[..., :3, 3:4] # (SB, 1, ray_batch_size, 3, 1)

    # Calculate relative pose
    R_t_transpose = R_t.permute(0, 1, 2, 4, 3) # (SB, 1, ray_batch_size, 3, 3)
    R_r = R_t_transpose @ R_s                  # (SB, NS, ray_batch_size, 3, 3)
    t_r = R_t_transpose @ (t_s - t_t)          # (SB, NS, ray_batch_size, 3, 1)

    # Concatenate
    relative_poses = torch.cat(
        [
            torch.cat([R_r, t_r], dim=-1), # (SB, NS, ray_batch_size, 3, 4)
            torch.tensor([0, 0, 0, 1], device=src_poses.device).view(1, 1, 1, 1, 4)
        ],
        dim=-2
    ) # (SB, NS, ray_batch_size, 4, 4)

    return relative_poses


class VanillaPixelnerfViewCombiner(nn.Module):

    def combine_interleaved(self, t, inner_dims=(1,), agg_type="average"):
        if len(inner_dims) == 1 and inner_dims[0] == 1:
            return t
        t = t.reshape(-1, *inner_dims, *t.shape[1:])
        if agg_type == "average":
            t = torch.mean(t, dim=1)
        elif agg_type == "max":
            t = torch.max(t, dim=1)[0]
        else:
            raise NotImplementedError("Unsupported combine type " + agg_type)
        return t

    def forward(self, x, combine_inner_dims, combine_type="average", **kwargs):
        """
        :param x (..., d_hidden)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1
        """
        return self.combine_interleaved(x, combine_inner_dims, combine_type)


class CamDistanceAngleErrorCombiner(nn.Module):
    """
    Weighs input views based on the distance and angle between the source and target views.
    source views that are closer to the target view are weighted more heavily. Angle is also
    taken into account.

    Angle vs Distance are weighted by alpha, 1-alpha respectively.
    """

    def __init__(self, alpha=0.5):
        assert(alpha >= 0 and alpha <= 1.0), "alpha must be between [0, 1]"
        self.alpha = alpha
        super().__init__()

    def forward(self, x, combine_inner_dims, combine_type="average", src_poses=None, target_poses=None, **kwargs):
        """
        :param x (..., d_hidden)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduce on axis 1
        :param src_poses (SB, NS, 4, 4) source poses
        :param target_poses (SB, ray_batch_size, 4, 4) target poses
        """

        # reshape for broadcasting
        SB, NS, _, _ = src_poses.shape
        _, ray_batch_size, _, _ = target_poses.shape
        src_poses = src_poses.reshape(SB, NS, 1, 4, 4)
        target_poses = target_poses.reshape(SB, 1, ray_batch_size, 4, 4)

        # extracy rotation and translation
        R_s = src_poses[..., :3, :3]     # (SB, NS, 1             , 3, 3)
        R_t = target_poses[..., :3, :3]  # (SB, 1 , ray_batch_size, 3, 3)
        t_s = src_poses[..., :3, 3:4]    # (SB, NS, 1             , 3, 1)
        t_t = target_poses[..., :3, 3:4] # (SB, 1 , ray_batch_size, 3, 1)

        # calculate distance and angle
        c_s = -R_s.permute(0, 1, 2, 4, 3) @ t_s # (SB, NS, 1, 3, 1)
        c_t = -R_t.permute(0, 1, 2, 4, 3) @ t_t # (SB, 1 , ray_batch_size, 3, 1)
        dist = torch.norm(c_s - c_t, dim=-2) # (SB, NS, ray_batch_size, 1)
        dist = dist[..., 0] # (SB, NS, ray_batch_size)
        angle = torch.acos(torch.sum(R_s[...,2,:]*R_t[...,2,:], dim=-1)) # (SB, NS, ray_batch_size)

        # calculate weights
        dist_weight  = 1 - dist  / torch.max(dist , dim=-1, keepdim=True)[0] # (SB, NS, ray_batch_size)
        angle_weight = 1 - angle / torch.max(angle, dim=-1, keepdim=True)[0] # (SB, NS, ray_batch_size)
        weight = self.alpha * dist_weight + (1 - self.alpha) * angle_weight # (SB, NS, ray_batch_size)

        # print('SB: ', SB)
        # print('NS: ', NS)
        # print('ray_batch_size: ', ray_batch_size)
        # print('angle.shape: ', angle.shape)
        # print('dist.shape: ', dist.shape)
        


        # print('x.shape: ', x.shape)
        x = x.reshape(-1, *combine_inner_dims, *x.shape[1:])
        # print('x.shape: ', x.shape)


        return torch.mean(x, dim=1)

    