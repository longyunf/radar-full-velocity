import numpy as np
import os
from os.path import join
import h5py
import torch
from torch.utils import data
from nuscenes.nuscenes import NuScenes

from gt_velocity import gt_box_key, cal_depthMap, cal_trans_matrix, cal_matrix_refSensor_to_global
from nb_utils import sparse_neighbor_connection


def get_intrinsic_matrix(nusc, cam_token):        
    cam_data = nusc.get('sample_data', cam_token)
    cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    return np.array( cs_rec['camera_intrinsic'] )

      
def cal_uv1(h, w, K, downsample_scale=4, y_cutoff=33):
    '''
    uv_map: h x w x 2
    '''
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map, y_map = x_map.astype('float32'), y_map.astype('float32')
    
    cx = cx / downsample_scale
    cy = cy / downsample_scale - y_cutoff
    f = f / downsample_scale
    
    u_map = (x_map - cx) / f
    v_map = (y_map - cy) / f
        
    uv_map = np.stack([u_map,v_map], axis=2)
    
    return uv_map


def cal_uv_translation(uv2, R, msk_uv2=None):
    '''
    inputs:
        uv2: (2 x h x w); full flow
        R: rotaion matrix (from u1,v1 -> u2,v2)
        msk_uv2: h x w
    output:
        uvt2: flow from translation inv(R)*t   
    '''    
    u2, v2 = uv2[0], uv2[1]

    R_inv = np.linalg.inv(R)
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = R_inv.flatten()
    ut = (u2*r11 + v2*r12 + r13) / (u2*r31 + v2*r32 +r33)
    vt = (u2*r21 + v2*r22 + r23) / (u2*r31 + v2*r32 +r33)
    
    if msk_uv2 is not None:
        ut = ut * msk_uv2
        vt = vt * msk_uv2
        
    uvt2 = np.stack([ut,vt])

    return uvt2


def init_data_loader(args, mode):
    
    if mode == 'train':
        batch_size = args.batch_size
        if args.no_data_shuffle:
            shuffle = False
        else:
            shuffle = True
    else:
        batch_size = args.test_batch_size
        shuffle = False
    
    nusc = NuScenes(version = args.version, dataroot = args.dir_nuscenes, verbose=False)
        
    args_dataset = {'path_data_file': args.path_data_file,
                    'mode': mode,
                    'nusc':nusc,
                    'nb':args.nb}
    args_data_loader = {'batch_size': batch_size,
                       'shuffle': shuffle,
                       'num_workers': args.num_workers}
    dataset = Dataset(**args_dataset)    
    data_loader = torch.utils.data.DataLoader(dataset, **args_data_loader)
    
    return data_loader
    

class Dataset(data.Dataset):     
    def __init__(self, path_data_file, mode, nusc, nb):                        
        data = h5py.File(path_data_file, 'r')[mode] 
        self.nusc = nusc
        self.im_list = data['im'][...]
        self.uv2_im_list = data['im_uv'][...].astype('f4')
        self.indices = data['indices']
        self.nb = nb
                           
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        
        sample_idx = self.indices[idx]
        gt = gt_box_key(self.nusc, sample_idx)
        rd = gt.radar              
        uv2_im = self.uv2_im_list[idx].astype('float32')           
        h, w = uv2_im.shape[:2]
        
        cam_data = self.nusc.get('sample_data', gt.cam_token) 
        cam_token2 = cam_data['next']        
        dt = (self.nusc.get('sample_data', cam_token2)['timestamp'] - cam_data['timestamp']) * 1e-6
               
        cam_token1 = gt.cam_token
        rd_token = gt.radar_token
                
        K = get_intrinsic_matrix(self.nusc, cam_token1)      
        uv1_im = cal_uv1(h, w, K, downsample_scale=4, y_cutoff=33)
        uv = np.concatenate([uv1_im, uv2_im], axis=2)
          
        T_c2r = cal_trans_matrix(self.nusc, cam_token1, rd_token)
        T_c2c = cal_trans_matrix(self.nusc, cam_token1, cam_token2)
        T_c2w = cal_matrix_refSensor_to_global(self.nusc, cam_token1)
                 
        depth_map, _, msk_nb_map, vx_nb_map, vy_nb_map, _, vx_gt_map, vy_gt_map = \
            cal_depthMap(rd, uv, T_c2r, T_c2c, T_c2w, dt, self.nb, downsample_scale=4, y_cutoff=33)
        
        error_map = ( (vx_nb_map - vx_gt_map[...,None])**2 + (vy_nb_map - vy_gt_map[...,None])**2 )**0.5
        
        error_map = error_map.transpose((2,0,1))     # (n,h,w)
        msk_nb_map = msk_nb_map.transpose((2,0,1))
                
        im1 = self.im_list[idx].astype('float32').transpose((2,0,1))/255   # (3,h,w)    
        R = T_c2c[:3,:3]
        
        uv1_im = uv1_im.transpose((2,0,1))  # (2,h,w)             
        uv2_im = uv2_im.transpose((2,0,1))  # (2,h,w)        
        
        d_radar = depth_map[None,...].astype('float32')   # (1,h,w)    
        
        scale_factor = 30
       
        uvt2_im = cal_uv_translation(uv2_im, R)        
        duv_im = (uvt2_im - uv1_im) * scale_factor                    
        
        data_in = np.concatenate((im1, uv1_im, duv_im, d_radar), axis=0)    # (8,h,w)
        
        sample = {'data_in': data_in,  'sample_idx': self.indices[idx], 'error': error_map, 'msk': msk_nb_map}
                           
        return sample


if __name__=='__main__':    
    this_dir = os.path.dirname(__file__)
    dir_data = join(this_dir, '..', 'data')  
    dir_nuscenes = join(dir_data, 'nuscenes')
    path_data_file = join(dir_data, 'prepared_data.h5')
    nb = sparse_neighbor_connection(*(4, 4, 10, 4))
    nusc = NuScenes(version = 'v1.0-trainval', dataroot = dir_nuscenes, verbose=False)
    
    args_dataset = {'path_data_file': path_data_file,
                    'mode': 'train',
                    'nusc': nusc,
                    'nb':nb}
    args_train_loader = {'batch_size': 6,
                         'shuffle': True,
                         'num_workers': 0}
  
    train_set = Dataset(**args_dataset)    
    train_loader = torch.utils.data.DataLoader(train_set, **args_train_loader)
    
    data_iterator = enumerate(train_loader)
    
    batch_idx, sample = next(data_iterator)
    
    print('batch_idx', batch_idx)
    print('data_in', sample['data_in'].shape, type(sample['data_in']),sample['data_in'].dtype)
    print('error', sample['error'].shape, type(sample['error']), sample['error'].dtype)
    print('msk', sample['msk'].shape, type(sample['msk']), sample['msk'].dtype)
