import argparse
import os
from os.path import join
import numpy as np
import h5py
from tqdm import tqdm
import skimage.io as io
import torch

def create_data_group(hf, mode, sample_indices, dir_label):       
    group = hf.create_group('%s' % mode)
        
    im_list = []
    for idx in tqdm(sample_indices, '%s:im' % mode):
        im1 = io.imread(join(dir_label, '%05d_im.jpg' % idx))
        im_list.append(im1)    
    group.create_dataset('im',data=np.array(im_list))
    del im_list
    
    uv2_im_list = []
    for idx in tqdm(sample_indices, '%s:uv2_im' % mode):
        uv2_im = np.load(join(dir_label, '%05d_im_uv.npy' % idx))
        uv2_im_list.append(uv2_im) 
    group.create_dataset('im_uv',data=np.array(uv2_im_list))
    del uv2_im_list
       
    group.create_dataset('indices',data=np.array(sample_indices))
    
        
if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)
    args = parser.parse_args()
    
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')
    
    dir_label = join(args.dir_data, 'prepared_data') 
    path_h5_file = join(args.dir_data, 'prepared_data.h5')

    train_sample_indices = torch.load(join(args.dir_data,'sample_split.tar'))['train_sample_indices']
    val_sample_indices = torch.load(join(args.dir_data,'sample_split.tar'))['val_sample_indices']
    test_sample_indices = torch.load(join(args.dir_data,'sample_split.tar'))['test_sample_indices']
    
    hf = h5py.File(path_h5_file, 'w')
        
    create_data_group(hf, 'train', train_sample_indices, dir_label)
    create_data_group(hf, 'val', val_sample_indices, dir_label)
    create_data_group(hf, 'test', test_sample_indices, dir_label)
    
    hf.close()
    
    