import sys
import os
from typing import Any
from torchsummary import summary

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import util
import warnings
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import tqdm
import csv


def extra_args(parser):
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="64",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=40,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    return parser


def move_to_device(device,*tensor_list):
    return [t.to(device=device) for t in tensor_list]


def preprocess_images(images):
    # use min and max per image
    min = images.min(dim=1,keepdim=True)[0]
    min =    min.min(dim=2,keepdim=True)[0]
    min =    min.min(dim=3,keepdim=True)[0]
    max = images.max(dim=1,keepdim=True)[0]
    max =    max.max(dim=2,keepdim=True)[0]
    max =    max.max(dim=3,keepdim=True)[0]
    images -= min
    images /= max
    images = images*2.0 - 1.0
    return images


def post_process_image(image,chan_process = 'max'):
    ''' image.shape = (C,H,W) '''
    if chan_process == 'first3':
        image = image[:3]
    else:
        # break up the channels into 3 sections
        n_chan = image.shape[0]
        chan_size = int(n_chan//3)
        chan_size = np.repeat(chan_size,3)
        n_current_chan = chan_size.sum()
        n_missing_chan = n_chan - n_current_chan
        chan_size[:n_missing_chan] += 1
        assert(chan_size.sum() == n_chan), 'chan_size.sum() != n_chan\n%d != %d'%(chan_size.sum(),n_chan)
        image = torch.split(image,chan_size.tolist(),dim=0) # [(C1,H,W),(C2,H,W),(C3,H,W)]

        # take max or mean of each section over the channel dimension
        if chan_process == 'max':
            image = [chan.max(dim=0,keepdim=True)[0] for chan in image] # [(1,H,W),(1,H,W),(1,H,W)]
        elif chan_process == 'mean':
            image = [chan.mean(dim=0,keepdim=True) for chan in image] # [(1,H,W),(1,H,W),(1,H,W)]
        else:
            raise ValueError('chan_process = %s is not supported'%chan_process)
        image = torch.cat(image,dim=0) # (3,H,W)

    if torch.any(image < 0):
        image -= image.min()
    image = image.detach().cpu().numpy()
    image = np.transpose(image,(1,2,0))
    image /= image.max()
    image *= 255
    image = image.astype(np.uint8)
    return image



def manual_shapenet_example(
    device=None,
    folder = os.path.join('/','home','berian','Documents','shapenet','cars_test'),
    object_id = '9f69ac0aaab969682a9eb0f146e94477',
    in_pose_ids = ['000101',],
    target_pose_id = '000102',
    ):
    '''
    Loads a specific example from shapenet

    Args:
        device: torch.device
        folder: str
        object_id: str
        in_pose_ids: list of str
        target_pose_id: str
    returns:
        in_images: (n_in_image,3,H,W)
        in_poses: (n_in_image,4,4)
        in_f: (n_in_image,)
        target_image: (3,H,W)
        target_pose: (4,4)
        target_f: ()
    '''
    # get focal length
    intrinsics_path = os.path.join(folder,object_id,'intrinsics.txt')
    with open(intrinsics_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        string_list = [row for row in reader]
    f = float(string_list[0][0])

    # get pose and image paths
    all_pose_ids = in_pose_ids + [target_pose_id,]
    num_pose = len(all_pose_ids)
    image_paths = [os.path.join(folder,object_id,'rgb',p+'.png') for p in all_pose_ids]
    pose_paths = [os.path.join(folder,object_id,'pose',p+'.txt') for p in all_pose_ids]

    # load images
    images = []
    for image_path in image_paths:
        image = plt.imread(image_path)
        image = image[:,:,:3] # remove alpha channel
        image = torch.tensor(image,dtype=torch.float32,device=device)
        image = image.permute(2,0,1).unsqueeze(0) # (1,3,H,W)
        images.append(image)
    images = torch.cat(images,dim=0) # (num_pose,3,H,W)

    # load pose matrices
    poses = []
    for pose_path in pose_paths:
        pose = np.loadtxt(pose_path)
        pose = torch.tensor(pose,dtype=torch.float32,device=device)
        pose = pose.reshape(1,4,4)
        poses.append(pose)
    poses = torch.cat(poses,dim=0) # (num_pose,4,4)

    # repeat focal length
    f = torch.tensor(f,dtype=torch.float32,device=device)
    f = f.repeat(num_pose) # (num_pose)

    # done
    return images[:-1],poses[:-1],f[:-1],images[-1],poses[-1],f[-1]





def original_code(
        args,
        conf,
):
    args.resume = True

    device = util.get_cuda(args.gpu_id[0])

    dset = get_split_dataset(
        args.dataset_format, args.datadir, want_split=args.split, training=False
    )

    data = dset[args.subset]
    data_path = data["path"]
    print("Data instance loaded:", data_path)

    images = data["images"]  # (NV, 3, H, W)

    poses = data["poses"]  # (NV, 4, 4)
    focal = data["focal"]
    if isinstance(focal, float):
        # Dataset implementations are not consistent about
        # returning float or scalar tensor in case of fx=fy
        focal = torch.tensor(focal, dtype=torch.float32)
    focal = focal[None]

    c = data.get("c")
    if c is not None:
        c = c.to(device=device).unsqueeze(0)

    NV, _, H, W = images.shape

    if args.scale != 1.0:
        Ht = int(H * args.scale)
        Wt = int(W * args.scale)
        if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
            warnings.warn(
                "Inexact scaling, please check {} times ({}, {}) is integral".format(
                    args.scale, H, W
                )
            )
        H, W = Ht, Wt

    net = make_model(conf["model"]).to(device=device)
    net.load_weights(args)

    renderer = NeRFRenderer.from_conf(
        conf["renderer"], lindisp=dset.lindisp, eval_batch_size=args.ray_batch_size,
    ).to(device=device)

    render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

    # Get the distance from camera to origin
    z_near = dset.z_near
    z_far = dset.z_far

    print("Generating rays")

    dtu_format = hasattr(dset, "sub_format") and dset.sub_format == "dtu"

    if dtu_format:
        print("Using DTU camera trajectory")
        # Use hard-coded pose interpolation from IDR for DTU

        t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
        pose_quat = torch.tensor(
            [
                [0.9698, 0.2121, 0.1203, -0.0039],
                [0.7020, 0.1578, 0.4525, 0.5268],
                [0.6766, 0.3176, 0.5179, 0.4161],
                [0.9085, 0.4020, 0.1139, -0.0025],
                [0.9698, 0.2121, 0.1203, -0.0039],
            ]
        )
        n_inter = args.num_views // 5
        args.num_views = n_inter * 5
        t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
        scales = np.array([2.0, 2.0, 2.0, 2.0, 2.0]).astype(np.float32)

        s_new = CubicSpline(t_in, scales, bc_type="periodic")
        s_new = s_new(t_out)

        q_new = CubicSpline(t_in, pose_quat.detach().cpu().numpy(), bc_type="periodic")
        q_new = q_new(t_out)
        q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
        q_new = torch.from_numpy(q_new).float()

        render_poses = []
        for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
            new_q = new_q.unsqueeze(0)
            R = util.quat_to_rot(new_q)
            t = R[:, :, 2] * scale
            new_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            new_pose[:, :3, :3] = R
            new_pose[:, :3, 3] = t
            render_poses.append(new_pose)
        render_poses = torch.cat(render_poses, dim=0)
    else:
        print("Using default (360 loop) camera trajectory")
        if args.radius == 0.0:
            radius = (z_near + z_far) * 0.5
            print("> Using default camera radius", radius)
        else:
            radius = args.radius

        # Use 360 pose sequence from NeRF
        render_poses = torch.stack(
            [
                util.pose_spherical(angle, args.elevation, radius)
                for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
            ],
            0,
        )  # (NV, 4, 4)

    render_rays = util.gen_rays(
        render_poses,
        W,
        H,
        focal * args.scale,
        z_near,
        z_far,
        c=c * args.scale if c is not None else None,
    ).to(device=device)
    # (NV, H, W, 8)

    focal = focal.to(device=device)

    source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
    NS = len(source)
    random_source = NS == 1 and source[0] == -1
    assert not (source >= NV).any()

    if renderer.n_coarse < 64:
        # Ensure decent sampling resolution
        renderer.n_coarse = 64
        renderer.n_fine = 128

    with torch.no_grad():
        print("Encoding source view(s)")
        if random_source:
            src_view = torch.randint(0, NV, (1,))
        else:
            src_view = source

        net.encode(
            images[src_view].unsqueeze(0),
            poses[src_view].unsqueeze(0).to(device=device),
            focal,
            c=c,
        )

        print("Rendering", args.num_views * H * W, "rays")
        all_rgb_fine = []
        for rays in tqdm.tqdm(
            torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)
        ):
            rgb, _depth = render_par(rays[None])
            all_rgb_fine.append(rgb[0])
        _depth = None
        rgb_fine = torch.cat(all_rgb_fine)
        # rgb_fine (V*H*W, 3)

        frames = rgb_fine.view(-1, H, W, 3)
    


def predict_novel_views(
        obj_folder,
        n_in_views,
        args,
        conf,
    ):
    '''
    goes through a given SRN object folder in the SRN dataset, and 
    generates the predicted novel view for each pose. The predicted 
    novel view uses 5 randomly selected source views.
    '''
    # load images and poses
    dataset_folder = args.datadir+'_test'
    full_path = os.path.join(dataset_folder,obj_folder)
    image_folder = os.path.join(full_path,'rgb')
    all_fname_prefixes = os.listdir(image_folder)
    all_fname_prefixes = [f.split('.')[0] for f in all_fname_prefixes]
    images, \
    poses, \
    f, \
    _,_,_ = manual_shapenet_example(folder=dataset_folder,
                    object_id=obj_folder,in_pose_ids=all_fname_prefixes)
    # images: (n_in_image,3,H,W)
    # poses: (n_in_image,4,4)
    # f: (n_in_image,)

    # initialize the novel view synthesizer
    nvs = NovelViewSynthesizer(args,conf)

    # loop over each pose
    for i in range(len(poses)):

        # get index of the random input views
        possible_i = list(range(len(poses)))
        possible_i.remove(i)
        random_i = np.random.choice(possible_i,n_in_views,replace=False)

        # get input poses and images
        in_poses = poses[random_i] # (n_in_image,4,4)
        in_images = images[random_i] # (n_in_image,3,H,W)
        in_f = f[0:1] # (1,)

        # get target pose and image
        target_pose = poses[i]
        target_image = images[i]
        target_f = f[i]

        # move to device
        in_images, in_poses, in_f, target_image, target_pose, target_f = move_to_device(
            nvs.device,in_images,in_poses,in_f,target_image,target_pose,target_f
        )

        # preprocess images
        images = preprocess_images(images)

        # use the nvs to predict the novel view
        frame = nvs(in_images,in_poses,target_pose)

        # post process and save the predicted image
        print('frames.shape',frames.shape)
        im = frames[0]
        im = im.permute(2,0,1) # (3,H,W)
        im = post_process_image(im)
        print('im.shape',im.shape)

        # create predicted images folder
        predicted_folder = os.path.join(full_path,'pixelnerf_predicted_images')
        if not os.path.exists(predicted_folder):
            os.makedirs(predicted_folder)

        # save image
        image_path = os.path.join(predicted_folder,all_fname_prefixes[i]+'.png')
        print('saving image to ',image_path)
        imageio.imwrite(image_path, im)







class NovelViewSynthesizer:
    '''
    contains the necessary information to generate novel views of an object
    '''
    def __init__(self,args,conf):

        self.args = args
        self.conf = conf
        
        # idk what this does
        args.resume = True

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = util.get_cuda(args.gpu_id[0])
        print('self.device: ',self.device)

        # render params
        self.focal = torch.tensor([131.25,],dtype=torch.float32,device=self.device)
        self.H = 128
        self.W = 128
        self.z_near = 0.8
        self.z_far = 1.8


        print(conf['model'])

        # make model
        print('Making model...')
        self.net = make_model(conf["model"])
        self.net.to(device=self.device)
        print('Loading weights...')
        self.net.load_weights(args)

        # make renderer
        print('Making renderer...')
        self.renderer = NeRFRenderer.from_conf(
            conf["renderer"], 
            lindisp=False, #dset.lindisp,
            eval_batch_size=args.ray_batch_size,
        ).to(device=self.device)
        self.render_par = self.renderer.bind_parallel(
            self.net,
            args.gpu_id,
            simple_output=True
        ).eval()



    def __call__(self,in_images,in_poses,target_pose):
        target_pose = target_pose.unsqueeze(0)

        # get rays
        render_rays = util.gen_rays(
            target_pose,
            self.W,
            self.H,
            self.focal,
            self.z_near,
            self.z_far,
            None,
        ).to(device=self.device)

        # encode input images
        self.net.encode(
            in_images,
            in_poses,
            self.focal,
            None,
        )

        # render
        all_rgb_fine = []
        for rays in tqdm.tqdm(
            torch.split(render_rays.view(-1, 8), self.args.ray_batch_size, dim=0)
        ):
            rgb, _depth = self.render_par(rays[None])
            all_rgb_fine.append(rgb[0])
        _depth = None
        rgb_fine = torch.cat(all_rgb_fine)
        
        # reshape and return
        frames = rgb_fine.view(-1, self.H, self.W, 3)
        frame = frames[0]
        return frame


if __name__ == '__main__':
    
    args, conf = util.args.parse_args(extra_args)


    # # lab computer 
    # dataset_folder = os.path.join('/','home','berian','Documents','shapenet','cars_test')
    # obj_folder = '9f69ac0aaab969682a9eb0f146e94477'

    # home computer
    dataset_folder = '/mnt/c/Users/Berian/Documents/Arizona/research/Mahalanobis/genvs/shapenet'
    obj_folder = '9eaafc3581357b05d52b599fafc842f'
    predict_novel_views(obj_folder,5,args,conf)
