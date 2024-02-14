import os
import time
import csv
from matplotlib import pyplot as plt
from training.augment import AugmentPipe
import torch
import numpy as np
import sys
import imageio
import PIL

from training.networks import GeNVSEncoder, BerianFeatureImageGenerator


def save_fig(fname,verbose=True,dpi='figure',folder = os.path.join('.','figures')):
    '''
    im sick of doing savefig(os.path.join(...)) clf close many times
    dpi=600 is good for a plotting a single (100,500) image
    '''
    if not os.path.exists(folder):
        print('creating folder \'%s\''%folder)
        os.mkdir(folder)
    if verbose:
        print('saving \'%s\'...'%fname)
    plt.savefig(os.path.join('.','figures',fname),bbox_inches='tight',dpi=dpi)
    plt.clf()
    plt.close()


def create_gif_from_pngs(header, folder_name=os.path.join('.','figures'), frame_duration=.5):
    '''
    writte by ChatGPT
    Creates a GIF from all PNG files in the folder that start with the given header.
    '''
    gif_path = os.path.join(folder_name, f'{header}.gif')
    print('making \'%s\'...'%gif_path)
    file_names = [file_name for file_name in os.listdir(folder_name) if file_name.startswith(header) and file_name.endswith('.png')]
    file_names.sort()
    images = [imageio.v2.imread(os.path.join(folder_name, file_name)) for file_name in file_names]
    try:
        imageio.mimsave(gif_path, images, duration=frame_duration)
    except(ValueError):
        print('WARNING: Cannot make \'%s\' because not all images are the same size.')


def shapenet_example(num_pose = 2, device=None, folder = os.path.join('/','home','berian','Documents','shapenet','cars_train')):
    '''
    Loads a random example from shapenet

    returns:
        images: (num_pose,3,H,W)
        poses: (num_pose,4,4)
        f: (num_pose,)
    '''
    sub_folders = os.listdir(folder)

    # get random sub folder
    sub_folder = np.random.choice(sub_folders)

    # get focal length
    intrinsics_path = os.path.join(folder,sub_folder,'intrinsics.txt')
    with open(intrinsics_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        string_list = [row for row in reader]
    f = float(string_list[0][0])

    # get file names
    fnames = os.listdir(os.path.join(folder,sub_folder,'rgb'))
    prefix = [fname[:-4] for fname in fnames]    

    # pick num_pose random poses without replacement
    prefix = np.random.choice(prefix,num_pose,replace=False)

    # get pose and image paths
    image_paths = [os.path.join(folder,sub_folder,'rgb',p+'.png') for p in prefix]
    pose_paths = [os.path.join(folder,sub_folder,'pose',p+'.txt') for p in prefix]

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
    return images,poses,f



def shapenet_batch(max_num_pose=3, batch_size = 3, device=None, same_pose_prob=0.0, folder = os.path.join('/','home','berian','Documents','shapenet','cars_train')):
    '''
    Loads a batch of random examples from shapenet

    returns:
        images: (n_in_image,3,H,W)
        poses: (n_in_image,4,4)
        f: (n_in_image,)
        view_start_stop: (batch_size,2,). example: ((0,3),(3,4),(4,5),...)
    '''
    # get each sample
    in_images          = []
    in_poses           = []
    in_f               = []
    out_images         = []
    out_poses          = []
    out_f              = []
    view_per_sample = []
    for _ in range(batch_size):

        num_pose = np.random.randint(max_num_pose)+2 # at least 2 poses, one for input, one for target
        image,pose,focal_length = shapenet_example(num_pose=num_pose,device=device,folder=folder)

        in_images.append(image[:-1])
        in_poses.append(pose[:-1])
        in_f.append(focal_length[:-1])
        view_per_sample.append(num_pose-1) # -1 because one pose is for target, the rest are for input

        # randomly choose input or target pose
        if np.random.rand() < same_pose_prob:
            out_images.append(image[:1])
            out_poses.append(pose[:1])
            out_f.append(focal_length[:1])
        else:
            out_images.append(image[-1:])
            out_poses.append(pose[-1:])
            out_f.append(focal_length[-1:])



    # concatenate along pose dimension
    in_images = torch.cat(in_images,dim=0) # (n_in_image,3,H,W)
    in_poses = torch.cat(in_poses,dim=0)   # (n_in_image,4,4)
    in_f = torch.cat(in_f,dim=0)           # (n_in_image,)
    out_images = torch.cat(out_images,dim=0) # (batch_size,3,H,W)
    out_poses = torch.cat(out_poses,dim=0)   # (batch_size,4,4)
    out_f = torch.cat(out_f,dim=0)           # (batch_size,)

    # compute 
    # - view_start_stop : start and stop index for the views of example in a batch (batch_size,2,). example: ((0,3),(3,4),(4,5),...)
    view_start_stop = []
    for i in range(batch_size):
        if i == 0:
            start = 0
        else:
            start = view_start_stop[i-1][1]
        stop = start + view_per_sample[i]
        view_start_stop.append([start,stop])
    view_start_stop = torch.tensor(view_start_stop,dtype=torch.int64,device=device)

    return in_images,in_poses,in_f,view_start_stop,out_images,out_poses,out_f


def post_process_image(image,chan_process = 'max'):
    ''' image.shape = (C,H,W) '''
    # mode for just keeping the first 3 channels
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

    # fit the image between 0~255
    if torch.any(image < 0):
        image -= image.min()
    image = image.detach().cpu().numpy()
    image = np.transpose(image,(1,2,0))
    image /= image.max()
    image *= 255
    image = image.astype(np.uint8)
    return image

def plot_torch_image(image):
    '''
    plots a torch image
    '''
    image = post_process_image(image)
    plt.imshow(image)



class ImageGenEncoderPair(torch.nn.Module):
    '''
    a pair of image generator and encoder
    '''
    def __init__(self, n_features=16, n_output_channels=None):
        super().__init__()

        # initialize the image generator
        self.image_gen = BerianFeatureImageGenerator(
            128, # image_width,
            128, # image_height,
            n_features, # n_features,
            # upsample_factor=2,
            # device=None,
            # samp_per_ray=64,
            negative_depth=False, # negative_depth=False,
        )
        self.device = self.image_gen.device

        # initialize the encoder
        self.encoder = GeNVSEncoder(volume_features=n_features).to(device=self.device)

        # initialize the one by one conv
        self.n_output_channels = n_output_channels
        if self.n_output_channels is not None:
            self.one_by_one_conv = torch.nn.Conv2d(n_features,n_output_channels,kernel_size=1,stride=1,padding=0).to(device=self.device)

        self.to(device=self.device)

    def forward(self, in_images, in_poses, in_f, vss, out_poses, out_f, stratified_sampling = True):
        in_feature_volumes = self.encoder(in_images) # (n_in_image,D,H,W)
        feature_images = self.image_gen(
            out_poses, # target_poses,
            out_f, # target_focal_lengths,
            in_poses, # input_poses,
            in_f, # input_focal_lengths,
            in_feature_volumes, # feature_volumes,
            vss, # view_start_stop,
            1.3, # near_distance,
            1.8, # far_distance,
            extrinsics_is_srn_pose = True,
            stratified_sampling=stratified_sampling,
            # nerf_batch_size=128,
            nerf_progress_bar=False,
            # mode='training',
        )

        if self.n_output_channels is not None:
            feature_images = self.one_by_one_conv(feature_images)

        return feature_images


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


def move_to_device(device,*tensor_list):
    return [t.to(device=device) for t in tensor_list]


def view_model_on_test( 
        model,
        max_num_pose = 1,
        batch_size = 20,
        same_pose_prob = 1,
        stratified_sampling = True,
        random_seed = 31832, # most detail i can see in stratified sampling. images are green
        plot_number = 0,
        feature_image_only = False,
    ):
    # make sure the plot number is 6 digits long
    num_str = str(plot_number).zfill(6)

    np.random.seed(random_seed)

    # get a batch
    in_images,in_poses,in_f,vss,out_images,out_poses,out_f = shapenet_batch(
        max_num_pose=max_num_pose,
        batch_size = batch_size,
        device=None,
        same_pose_prob=same_pose_prob,
        folder = os.path.join('/','home','berian','Documents','shapenet','cars_test'),
    )

    # move to device
    in_images,in_poses,in_f,vss,out_images,out_poses,out_f = move_to_device(
        model.device,
        in_images,in_poses,in_f,vss,out_images,out_poses,out_f
    )

    # preprocess images
    in_images = preprocess_images(in_images)
    out_images = preprocess_images(out_images)

    # forward pass
    feature_images = model(
        in_images, # in_images,
        in_poses, # in_poses,
        in_f, # in_f,
        vss, # vss,
        out_poses, # out_poses,
        out_f, # out_f,
        stratified_sampling=stratified_sampling,
    )

    # plot each batch point
    im_num = 0
    for start,stop in vss:

        if feature_image_only:
            image = post_process_image(feature_images[im_num])
            if not os.path.exists(os.path.join('.','figures')):
                os.mkdir(os.path.join('.','figures'))
            PIL.Image.fromarray(image).save(os.path.join('.','figures','feature_image_only_sample%d_batch%s.png'%(im_num,num_str)))

        else:

            n_images = stop-start
            n_images = n_images.item()

            input_images = in_images[start:stop]

            plot_col = 2
            plot_row = int(np.ceil((n_images+2)/2))

            for i in range(n_images):
                plt.subplot(plot_row,plot_col,i+1)
                plot_torch_image(input_images[i])
                plt.title('input image %d'%i)

            plt.subplot(plot_row,plot_col,n_images+1)
            plot_torch_image(out_images[im_num])
            plt.title('target image')

            plt.subplot(plot_row,plot_col,n_images+2)
            plot_torch_image(feature_images[im_num])
            plt.title('feature image')

            save_fig('feature_image_sample%d_batch%s.png'%(im_num,num_str) , dpi = 300*n_images)

        im_num += 1


def train_on_batch( model,
                    optim,
                    stratified_sampling = True,
                    batch_size = 4,
    ):
    # get a train batch
    in_images,in_poses,in_f,vss,out_images,out_poses,out_f = shapenet_batch(
        max_num_pose=3,# max_num_pose=max_num_pose,
        batch_size=batch_size,# batch_size = batch_size,
        # device=None,
        same_pose_prob=0.0,# same_pose_prob=same_pose_prob,
        folder = os.path.join('/','home','berian','Documents','shapenet','cars_train'),
    )

    # move to device
    in_images,in_poses,in_f,vss,out_images,out_poses,out_f = move_to_device(
        model.device,
        in_images,in_poses,in_f,vss,out_images,out_poses,out_f
    )
    
    # forward pass
    feature_images = model(
        in_images, # in_images,
        in_poses, # in_poses,
        in_f, # in_f,
        vss, # vss,
        out_poses, # out_poses,
        out_f, # out_f,
        stratified_sampling=stratified_sampling,
    ) # (batch_size,3,H,W)

    # backprop on mse loss
    loss = torch.nn.functional.mse_loss(feature_images[:,:3],out_images)
    print('train loss: ',loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()


def train_fimage_model(
        # random_seed = np.random.randint(100000)
        random_seed = 5831,
        batch_size = 1,
        n_view = 4,
        view_period = 10,
        save_period = 10000,
        learn_rate = 1e-3,
    ):
    # make the model
    print('Random Seed: ', random_seed)
    torch.manual_seed(random_seed)
    model = ImageGenEncoderPair(
        # n_features=3,
        n_features=16,
        n_output_channels=3,
    )

    # load the model's most recent weights
    # TODO: this

    # make the optimizer
    optim = torch.optim.Adam(model.parameters(),lr=learn_rate)

    # view the untrained model on some of the test set
    # include the input and target views in the plot
    view_model_on_test(
        model,
        max_num_pose = 3,
        batch_size = n_view,
        same_pose_prob = 0.5,
        stratified_sampling = False,
        random_seed = random_seed,
        plot_number = 0,
        feature_image_only = False,
    )

    # train loop
    batch_count = 0
    while True:

        # look at the model's results on the test set
        if batch_count % view_period == 0:
            view_model_on_test(
                model,
                max_num_pose = 3,
                batch_size = n_view,
                same_pose_prob = 0.0,
                stratified_sampling = False,
                random_seed = random_seed,
                plot_number = batch_count,
                feature_image_only = True,
            )

        # train on a batch
        print('training on batch %d...'%batch_count)
        tick = time.time()
        train_on_batch(
            model,
            optim,
            stratified_sampling = True,
            batch_size=batch_size,
        )
        tock = time.time()
        print('Batch %d took %f seconds'%(batch_count,tock-tick))

        # save the model's encoder and mlp every save_period batches
        if batch_count % save_period == 0:
            print('saving model...')
            if os.path.exists(os.path.join('.','models')) == False:
                os.mkdir(os.path.join('.','models'))
            save_path = os.path.join('.','models','fimage_encoder_%d.pth'%batch_count)
            torch.save(model.encoder.state_dict(),save_path)
            save_path = os.path.join('.','models','fimage_mlp_%d.pth'%batch_count)
            torch.save(model.image_gen.mlp.state_dict(),save_path)

        batch_count += 1



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


def view_model_on_example(
    in_images,
    in_poses,
    in_f,
    target_image,
    target_pose,
    target_f,
    model,
    plot_suffix = '',
    ):
    # useful variables
    n_in_image = in_images.shape[0]
    n_plot = n_in_image + 2

    # prepare stuff for the model
    in_images,in_poses,in_f,target_image,target_pose,target_f = move_to_device(
        model.device,
        in_images,in_poses,in_f,target_image,target_pose,target_f
    )
    in_images = preprocess_images(in_images)
    target_pose = target_pose.unsqueeze(0)
    target_f = target_f.unsqueeze(0)
    view_start_stop = torch.tensor([[0,n_in_image]],dtype=torch.int64,device=model.device)

    # forward pass
    feature_image = model(
        in_images, # in_images,
        in_poses, # in_poses,
        in_f, # in_f,
        view_start_stop, # vss,
        target_pose, # out_poses,
        target_f, # out_f,
        stratified_sampling=False,
    ) # (1,3,H,W)
    
    # plot input images
    for i in range(n_in_image):
        cam_center = in_poses[i,:3,3].detach().cpu().numpy()
        plt.subplot(1,n_plot,i+1)
        im = post_process_image(in_images[i])
        print('im.shape: ',im.shape)
        plt.imshow(im)
        plt.title('input image %d\nc=%s'%(i,str(cam_center)))

    # plot target image
    cam_center = target_pose[0,:3,3].detach().cpu().numpy()
    plt.subplot(1,n_plot,n_in_image+1)
    im = post_process_image(target_image)
    plt.imshow(im)
    plt.title('target image\nc=%s'%(str(cam_center)))

    # plot feature image
    plt.subplot(1,n_plot,n_in_image+2)
    im = post_process_image(feature_image[0])
    plt.imshow(im)
    plt.title('feature image')

    # save
    save_fig('feature_image_example_%s.png'%plot_suffix,dpi=300*n_plot)


def view_manual_example(
        folder = os.path.join('/','home','berian','Documents','shapenet','cars_test'),
        object_id = '9f69ac0aaab969682a9eb0f146e94477',
        in_pose_ids = ['000100',],
        target_pose_id = '000100',
        seed = 5831,
        # seed = np.random.randint(10000),
    ):

    # make the model
    print('Random Seed: ', seed)
    torch.manual_seed(seed)
    model = ImageGenEncoderPair(
        # n_features=3,
        n_features=16,
        n_output_channels=3,
    )

    # # train on a batch
    # optim = torch.optim.Adam(model.parameters(),lr=.1)
    # for i in range(5):
    #     train_on_batch(
    #         model,
    #         optim,
    #         stratified_sampling = True,
    #         batch_size=4,
    #     )

    # view the untrained model on some of the test set
    # include the input and target views in the plot
    in_pose_ids = (100+np.arange(-10,10)).tolist()
    in_pose_ids = ['%06d'%i for i in in_pose_ids]
    for in_pose_id in in_pose_ids:
        in_images,in_poses,in_f,target_image,target_pose,target_f = manual_shapenet_example(
            device=model.device,
            folder=folder,
            object_id=object_id,
            in_pose_ids=[in_pose_id,],
            target_pose_id=target_pose_id,
        )
        view_model_on_example(
            in_images,
            in_poses,
            in_f,
            target_image,
            target_pose,
            target_f,
            model,
            plot_suffix=str(in_pose_id),
        )

    # in_images,in_poses,in_f,target_image,target_pose,target_f = manual_shapenet_example(
    #     device=model.device,
    #     folder=folder,
    #     object_id=object_id,
    #     in_pose_ids=in_pose_ids,
    #     target_pose_id=target_pose_id,
    # )
    # view_model_on_example(
    #     in_images,
    #     in_poses,
    #     in_f,
    #     target_image,
    #     target_pose,
    #     target_f,
    #     model,
    # )




def create_training_gifs(
        sample_nums = np.arange(10),
        folder_name = os.path.join('.','figures'),
        fname_prefix = 'feature_image_only_sample',
        fname_suffix = '_batch',
        frame_duration = 0.5,
        batch_period = 100,
    ):
    '''
    Creates the gifs of the pseudo pixelnerf model training,

    This function takes all the images for a a particular sample, and creates a gif of the images that are a multiple of the batch_period.
    '''
    # get all filenames in the folder
    all_fnames = os.listdir(folder_name)

    # loop over every example
    for i in sample_nums:
        
        # get the input filenames
        input_fnames = []
        this_fname_prefix = '%s%d%s'%(fname_prefix,i,fname_suffix)
        gif_path = os.path.join(folder_name, f'{this_fname_prefix}.gif')
        print('making \'%s\'...'%gif_path)
        for fname in all_fnames:
            if fname.startswith(this_fname_prefix):
                input_fnames.append(fname)

        # get the batch numbers (the last 6 characters of the filename before the .png)
        batch_nums = [fname[-10:-4] for fname in input_fnames]
        # throw away the strings that are not numbers
        batch_nums = [int(bn) for bn in batch_nums if bn.isdigit()]

        # get the batch numbers that are multiples of batch_period
        batch_nums = [bn for bn in batch_nums if bn % batch_period == 0]

        # sort the batch numbers
        batch_nums.sort()

        # get the filenames for the batch numbers that are multiples of batch_period
        input_fnames = [this_fname_prefix + '%06d.png'%bn for bn in batch_nums]

        # make the gif
        images = [imageio.v2.imread(os.path.join(folder_name, file_name)) for file_name in input_fnames]
        try:
            imageio.mimsave(gif_path, images, duration=frame_duration)
        except(ValueError):
            print('WARNING: Cannot make \'%s\' because not all images are the same size.')

        # house keeping
        del images


if __name__ == '__main__':


    # train_fimage_model()

    # create_training_gifs(batch_period=300)
    # sys.exit()
    images,poses,f = shapenet_example(num_pose=1)

    image = images[0]
    pose = poses[0]
    f = f[0]

    print(image.shape)
    print(pose.shape)
    print(f.shape)

    print('Pose matrix: \n', pose)

    c = pose[:3,3]
    print('c: \n', c)

    mag_c = torch.norm(c)
    print('mag_c: \n', mag_c)

    homo_0 = torch.tensor([0,0,0,1],dtype=torch.float32)
    homo_0 = homo_0.reshape(4,1)


    





