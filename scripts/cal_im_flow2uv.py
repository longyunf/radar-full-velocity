import argparse
import os
from os.path import join
import numpy as np
from tqdm import tqdm
import torch
from nuscenes.nuscenes import NuScenes


def get_intrinsic_matrix(nusc, cam_token):        
    cam_data = nusc.get('sample_data', cam_token)
    cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    return np.array( cs_rec['camera_intrinsic'] )


def flow2uv_full(flow, K):
    '''
    uv_map: h x w x 2    
    '''
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    
    h,w = flow.shape[:2]
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map, y_map = x_map.astype('float32'), y_map.astype('float32')
    x_map += flow[..., 0]
    y_map += flow[..., 1]
    
    u_map = (x_map - cx) / f
    v_map = (y_map - cy) / f
    
    uv_map = np.stack([u_map,v_map], axis=2)
    
    return uv_map


def downsample_flow(flow_full, downsample_scale, y_cutoff):
    H, W, nc = flow_full.shape       
    h = int( H / downsample_scale )
    w = int( W / downsample_scale )
    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    
    x_map_old = np.round( np.clip( x_map * downsample_scale, 0, W-1) ).astype(int).ravel()
    y_map_old = np.round( np.clip( y_map * downsample_scale, 0, H-1) ).astype(int).ravel()
    
    flow_list = []
    for i in range(nc):
        flow_list.append(flow_full[y_map_old, x_map_old, i])
    
    flow = np.stack(flow_list, axis=1)    
    flow = np.reshape(flow, (h,w,-1))       
    flow = flow[y_cutoff:,...]
    
    return flow


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, help='data directory')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='dataset split')
    
    args = parser.parse_args()
        
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = os.path.join(this_dir, '..', 'data')
            
    dir_nuscenes = join(args.dir_data, 'nuscenes')    
    out_dir = join(args.dir_data, 'prepared_data')
    
    nusc = NuScenes(args.version, dataroot = dir_nuscenes, verbose=False)
        
    downsample_scale = 4
    y_cutoff = 33
        
    sample_indices = torch.load(join(args.dir_data,'sample_split.tar'))['all_indices'] 
         
    ct = 0         
    for sample_idx in tqdm(sample_indices):
        
        f_flow = join(out_dir, '%05d_full_flow.npy' % sample_idx)        
        flow = np.load(f_flow)
        
        cam_token = nusc.sample[sample_idx]['data']['CAM_FRONT']
        
        K = get_intrinsic_matrix(nusc, cam_token)
                        
        flow_downsampled = downsample_flow(flow, downsample_scale, y_cutoff)
        flow_downsampled /= downsample_scale
        
        uv_map = flow2uv_full(flow, K)
        uv_map = downsample_flow(uv_map, downsample_scale, y_cutoff)
                
        np.save(f_flow[:-13] + 'im_uv.npy', uv_map)       
        np.save(f_flow[:-13] + 'flow.npy', flow_downsampled)
        