import torch
import numpy as np


def fourier_features(images, n_features, dim=1):
    # intialize the functions for the features
    feature_functions = [
        lambda x: x,
        lambda x: torch.sin(1 * np.pi * x),
        lambda x: torch.cos(1 * np.pi * x),
        lambda x: torch.sin(2 * np.pi * x),
        lambda x: torch.cos(2 * np.pi * x),
        lambda x: torch.sin(4 * np.pi * x),
    ]

    # get the features
    features = [f(images) for f in feature_functions]
    features = torch.cat(features, dim=dim)

    # only keep the first n_features channels
    features = features[:,:,:,:n_features]
    return features


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