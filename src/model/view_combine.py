import sys
from torch import nn
import torch
from util import plt, save_fig, combine_interleaved
from .code import PositionalEncoding

#  import torch_scatter
import torch.autograd.profiler as profiler


def get_combine_module(combine_type, positional_encoder=None):
    """
    return a view combiner module based on the combine_type
    """

    combine_str_to_module = {
        "average": "VanillaPixelnerfViewCombiner",
        "max": "VanillaPixelnerfViewCombiner",
    }

    if "error" in combine_type:
        return CamDistanceAngleErrorCombiner(alpha=float(combine_type.split("error")[1]))

    elif combine_type == "cross_attention":
        return CrossAttentionCombiner(positional_encoder=positional_encoder)

    elif combine_type == "learned_cross_attention":
        return CrossAttentionCombiner(positional_encoder=positional_encoder, learned_attention=True)

    elif combine_type in combine_str_to_module:
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
    :param target_poses (SB, B', 4, 4) target poses
    :return (SB, NS, B', 4, 4) relative poses
    """
    # Reshape for broadcasting
    SB, NS, _, _ = src_poses.shape
    _, Bp, _, _ = target_poses.shape
    src_poses = src_poses.reshape(SB, NS, 1, 1, 4, 4)
    target_poses = target_poses.reshape(SB, 1, Bp, 1, 4, 4)

    # Extract rotation and translation
    R_s = src_poses[..., :3, :3]     # (SB, NS, 3, 3)
    R_t = target_poses[..., :3, :3]  # (SB, 1, B', 3, 3)
    t_s = src_poses[..., :3, 3:4]    # (SB, NS, 3, 1)
    t_t = target_poses[..., :3, 3:4] # (SB, 1, B', 3, 1)

    # Calculate relative pose
    R_t_transpose = R_t.permute(0, 1, 2, 4, 3) # (SB, 1, B', 3, 3)
    R_r = R_t_transpose @ R_s                  # (SB, NS, B', 3, 3)
    t_r = R_t_transpose @ (t_s - t_t)          # (SB, NS, B', 3, 1)

    # Concatenate
    relative_poses = torch.cat(
        [
            torch.cat([R_r, t_r], dim=-1), # (SB, NS, B', 3, 4)
            torch.tensor([0, 0, 0, 1], device=src_poses.device).view(1, 1, 1, 1, 4)
        ],
        dim=-2
    ) # (SB, NS, B', 4, 4)

    return relative_poses


class VanillaPixelnerfViewCombiner(nn.Module):

    def forward(self, x, combine_inner_dims, combine_type="average", **kwargs):
        """
        :param x (..., d_hidden)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1
        """
        return combine_interleaved(x, combine_inner_dims, combine_type)


class CamDistanceAngleErrorCombiner(nn.Module):
    """
    Weighs input views based on the distance and angle between the source and target views.
    source views that are closer to the target view are weighted more heavily. Angle is also
    taken into account.

    Angle vs Distance are weighted by alpha, 1-alpha respectively.
    """

    def __init__(self, alpha=0.5, epsilon=0.001):
        super().__init__()
        assert(alpha >= 0 and alpha <= 1.0), "alpha must be between [0, 1]"
        assert(epsilon >= 0), "epsilon must be non-negative"
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x, combine_inner_dims, combine_type="average", src_poses=None, target_poses=None, **kwargs):
        """
        poses are the srn_cars poses, not extrinsics
        [R.T | c], not [R | t]

        x's shape is (SB*NS*B'*K,H)
        combine_inner_dims is (NS,B'*K)

        SB = number of objects
        NS = number of source views per object
        B' = number of target rays per object
        K  = number of points per ray
        B  = B' * K
        H  = hidden dimension of the mlp
        
        :param src_poses (SB, NS, 4, 4) source poses
        :param target_poses (SB, B', 4, 4) target poses
        :param x (SB*NS*B'*K, H)
        :param combine_inner_dims tuple with (NS, B'*K)
        """
        # get shape information
        SB, NS, _, _ = src_poses.shape
        if NS == 1: # if only one source view, don't need to combine
            return combine_interleaved(x, combine_inner_dims, "average")
        _, Bp, _, _ = target_poses.shape
        K = combine_inner_dims[1] // Bp
        H = x.shape[-1]

        # extract view direction and camera center
        d_t = target_poses[..., :3, 2] # (SB, B', 3)
        d_s = src_poses[..., :3, 2]    # (SB, NS, 3)
        c_s = src_poses[..., :3, 3]    # (SB, NS, 3)
        c_t = target_poses[..., :3, 3] # (SB, B', 3)

        # calculate relative distance
        a = c_s.reshape(SB, NS, 1, 3) # (SB, NS, 1 , 3)
        b = c_t.reshape(SB, 1, Bp, 3) # (SB, 1 , B', 3)
        c = a - b                     # (SB, NS, B', 3)
        dist = torch.norm(c, dim=-1)  # (SB, NS, B')

        # calculate relative angle
        a = d_s/2*torch.norm(d_s, dim=-1, keepdim=True) # (SB, NS, 3)
        b = d_t/2*torch.norm(d_t, dim=-1, keepdim=True) # (SB, B', 3)
        a = a.reshape(SB, NS, 1, 3) # (SB, NS, 1, 3)
        b = b.reshape(SB, 1, Bp, 3) # (SB, 1, B', 3)
        c = a * b                   # (SB, NS, B', 3)
        angle = torch.acos(torch.sum(c, dim=-1)) # (SB, NS, B')

        # place the angle in the range [0, pi]
        angle = torch.remainder(angle, 2 * torch.pi)  # Step 1: Normalize to [0, 2*pi)
        angle[angle >= torch.pi] -= 2 * torch.pi      # Step 2: Adjust values >= pi to be in the range [-pi, pi)
        angle = torch.abs(angle)                      # Step 3: Take the absolute value

        # make sure there are no negative  or nan values
        assert(torch.all(torch.isfinite(dist)))       , "dist should be finite"
        assert(torch.all(torch.isfinite(angle)))      , "angle should be finite"
        assert(torch.all(torch.isnan(dist) == False)) , "dist should not be nan"
        assert(torch.all(torch.isnan(angle) == False)), "angle should not"
        assert(torch.all(dist >= 0))                  , "dist should be non-negative"
        assert(torch.all(angle >= 0))                 , "angle should be non-negative"
        assert(torch.all(angle <= torch.pi))          , "angle should be less than pi"

        # calculate weights
        dist_weight  = torch.softmax(-dist , dim=1) # (SB, NS, B')
        angle_weight = torch.softmax(-angle, dim=1) # (SB, NS, B')
        weight = self.alpha * dist_weight + (1 - self.alpha) * angle_weight  # (SB, NS, B')

        # apply weights
        weight = weight.reshape(SB, NS, Bp, 1, 1)
        x = x.reshape(SB, NS, Bp, K, H)
        x = x * weight          # (SB, NS, B', K, H)
        x = torch.sum(x, dim=1) # (SB, B', K, H)

        # reshape to expected output shape
        x = x.reshape(SB, Bp*K, H)
        return x


class CrossAttentionCombiner(nn.Module):
    """
    Uses cross attention between target pose and source poses to calculate weights for combining source views.
    Has the option of using or not using learned weights.

    Query: encoded TARGET camera center concatenated with unencoded direction of view
    Key: encoded SOURCE camera center concatenated with unencoded direction of view
    Value: hidden state of the MLP for source views
    """
    def __init__(self, positional_encoder=None, learned_attention = False, **kwargs):
        super().__init__(**kwargs)

        self.positional_encoder = positional_encoder
        self.learned_attention = learned_attention

        # initialize learned attention layers
        if self.learned_attention:
            self.attention_dim = self.positional_encoder.d_out + 3
            self.query_layer = nn.Linear(self.attention_dim, self.attention_dim)
            self.key_layer = nn.Linear(self.attention_dim, self.attention_dim)
        

    def forward(self, x, combine_inner_dims, combine_type="average", src_poses=None, target_poses=None, **kwargs):
        """
        x's shape is (SB*NS*B'*K,H)
        combine_inner_dims is (NS,B'*K)

        SB = number of objects
        NS = number of source views per object
        B' = number of target rays per object
        K  = number of points per ray
        B  = B' * K
        H  = hidden dimension of the mlp
        
        :param src_poses (SB, NS, 4, 4) source poses
        :param target_poses (SB, B', 4, 4) target poses
        :param x (SB*NS*B'*K, H)
        :param combine_inner_dims tuple with (NS, B'*K)
        """

        # get shape information
        SB, NS, _, _ = src_poses.shape
        if NS == 1: # if only one source view, don't need to combine
            return combine_interleaved(x, combine_inner_dims, "average")
        _, Bp, _, _ = target_poses.shape
        K = combine_inner_dims[1] // Bp
        H = x.shape[-1]

        # get camera centers and directions of view
        d_s = src_poses[..., :3, 2]     # (SB, NS, 3)
        d_t = target_poses[..., :3, 2]  # (SB, B', 3)
        c_s = src_poses[..., :3, 3]     # (SB, NS, 3)
        c_t = target_poses[..., :3, 3]  # (SB, B', 3)

        # calculate queries and keys
        coded_c_t = self.positional_encoder(c_t.reshape(-1,3)) # (SB*B', A-3) where A := self.attention_dim
        coded_c_s = self.positional_encoder(c_s.reshape(-1,3)) # (SB*NS, A-3)
        q = torch.cat([coded_c_t, d_t.reshape(-1,3)], dim=-1).reshape(SB,Bp,-1) # (SB, B', A)
        k = torch.cat([coded_c_s, d_s.reshape(-1,3)], dim=-1).reshape(SB,NS,-1) # (SB, NS, A)

        # use learned attention weights if needed
        if self.learned_attention:
            print("self.query_layer.weight: ", self.query_layer.weight)
            print("self.query_layer.bias: ", self.query_layer.bias)
            q = self.query_layer(q)
            k = self.key_layer(k)

        # calculate attention weights
        k_T = k.permute(0, 2, 1)      # (SB, A , NS)
        Wp = q @ k_T                  # (SB, B', NS)
        W = torch.softmax(Wp, dim=-1) # (SB, B', NS)
        W = W.permute(0, 2, 1)        # (SB, NS, B')
        
        # apply weights
        W = W.reshape(SB, NS, Bp, 1, 1)
        x = x.reshape(SB, NS, Bp, K, H)
        x = x * W               # (SB, NS, B', K, H)
        x = torch.sum(x, dim=1) # (SB, B', K, H)

        # reshape to expected output shape
        x = x.reshape(SB, Bp*K, H) # (SB, B*K, H)
        return x
