import os
from os.path import join
import numpy as np
import argparse
import skimage.io as io
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import torch
from nuscenes.utils.data_classes import LidarPointCloud

import _init_paths
from gt_velocity import gt_box_key, cal_trans_matrix, proj2im, flow2uv_map


def downsample_coord(xi_list, yi_list, downsample_scale=4, y_cutoff=33):
    
    h_im, w_im = 900, 1600
    h_new = int( h_im / downsample_scale )
    w_new = int( w_im / downsample_scale ) 
            
    xi_list_new = (xi_list + 0.5) / downsample_scale - 0.5
    yi_list_new = (yi_list + 0.5) / downsample_scale - 0.5
               
    xi_list_new = np.clip(xi_list_new, 0, w_new - 1)     
    yi_list_new = np.clip(yi_list_new, 0, h_new - 1) - y_cutoff
    
    return xi_list_new, yi_list_new


def upsample_coord(xi_list, yi_list, downsample_scale=4, y_cutoff=33):
    
    msk = xi_list == -1
    
    h_im, w_im = 900, 1600
    
    xi_list_new = xi_list
    yi_list_new = yi_list + y_cutoff
    
    xi_list_new *= downsample_scale
    yi_list_new *= downsample_scale
                   
    xi_list_new = np.clip(xi_list_new, 0, w_im-1)    
    yi_list_new = np.clip(yi_list_new, 0, h_im-1)
    
    xi_list_new[msk] = -1
    yi_list_new[msk] = -1
    
    return xi_list_new, yi_list_new


def correct_coord(xi_list, yi_list, rd_offset):    
            
    xi_list_new = []
    yi_list_new = []
    
    for xi, yi in zip(xi_list, yi_list):
        x_one, y_one = int(round( xi )), int(round( yi ))       
        dx, dy = rd_offset[y_one, x_one]
        if [dx, dy] != [-1000,-1000]:
            x_new, y_new = x_one + dx, y_one + dy           
        else:
            x_new, y_new = -1, -1
        xi_list_new.append(x_new)
        yi_list_new.append(y_new)
            
    xi_list_new = np.array(xi_list_new)
    yi_list_new = np.array(yi_list_new)
    
    return xi_list_new, yi_list_new


def get_im_pair(nusc, cam_token):
    cam_data = nusc.get('sample_data', cam_token) 
    K = np.array( nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic'] )
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
       
    cam_path = join(nusc.dataroot, cam_data['filename'])
    im1 = io.imread(cam_path)
    
    cam_token2 = cam_data['next']
    cam_data2 = nusc.get('sample_data', cam_token2)
    cam_path2 = join(nusc.dataroot, cam_data2['filename'])
    im2 = io.imread(cam_path2)
    
    dt = (cam_data2['timestamp'] - cam_data['timestamp']) * 1e-6
    
    return im1, im2, f, cx, cy, dt, cam_token2
    
       
def cal_full_v_in_radar(vx, vy, d, u1, v1, u2, v2, T_c2r, T_c2c, dt):    
    # output in radar coordinates              
     r11, r12, r13 = T_c2r[0,:3]
     r21, r22, r23 = T_c2r[1,:3]
     
     ra11, ra12, ra13, btx = T_c2c[0,:]
     ra21, ra22, ra23, bty = T_c2c[1,:]
     ra31, ra32, ra33, btz = T_c2c[2,:]
     
     A = np.array([[ra11-u2*ra31, ra12-u2*ra32, ra13-u2*ra33], \
                   [ra21-v2*ra31, ra22-v2*ra32, ra23-v2*ra33], \
                   [r11*vx+r21*vy, r12*vx+r22*vy, r13*vx+r23*vy]] )
         
     b = np.array([[((ra31*u1+ra32*v1+ra33)*u2-(ra11*u1+ra12*v1+ra13))*d+u2*btz-btx],\
                   [((ra31*u1+ra32*v1+ra33)*v2-(ra21*u1+ra22*v1+ra23))*d+v2*btz-bty],\
                   [(vx**2 + vy**2)*dt]])
         
     x = np.squeeze( np.dot( np.linalg.inv(A), b ) )
     
     vx_c, vy_c, vz_c = x[0]/dt, x[1]/dt, x[2]/dt
             
     vr = np.squeeze( np.dot(T_c2r[:3,:3], np.array([[vx_c], [vy_c], [vz_c]])) )
             
     vx_f, vy_f, vz_f = vr[0], vr[1], vr[2]
     
     return vx_f, vy_f, vz_f


def decompose_v(vx, vy, x, y):
    '''
    Decompose full velocity into radial and tengential components
    inputs:
        x,y: radar coordinates
        vx, vy: full velocity    
    outputs:
        radial_v: np.array([vx_radial, vy_radial])
        tangent_v: np.array([vx_tangent, vy_tangent])
    '''
    radial_v = (vx * x + vy * y) * np.array([x,y]) / (x**2 + y**2)
    tangent_v = np.array([vx,vy]) - radial_v
    
    return radial_v, tangent_v


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)  
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='dataset split')
    
    args = parser.parse_args()    
      
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')
     
    args.dir_nuscenes = join(args.dir_data, 'nuscenes')
    dir_files = join(args.dir_data, 'prepared_data')

    sample_indices = torch.load(join(args.dir_data,'sample_split.tar'))['test_sample_indices']
    nusc = NuScenes(args.version, dataroot = args.dir_nuscenes, verbose=False) 
      
    error_list = []
    error_radial_list = []
    error_tan_list = []

    error_list2 = []
    error_radial_list2 = []
    error_tan_list2 = []

    N = len(sample_indices)
    d_min = 0; d_max = float('inf')  
            
    for idx in tqdm(range(N)):    
        sample_idx = sample_indices[idx]
        cam_token1 = nusc.sample[sample_idx]['data']['CAM_FRONT']
        im1, im2, f, cx, cy, dt, cam_token2 = get_im_pair(nusc, cam_token1)
        
        gt = gt_box_key(nusc, sample_idx) 
        rd_token = gt.radar_token
        
        gt_rd = gt.radar
        x_list, y_list, vx_list, vy_list, msk_gt = gt_rd['x'], gt_rd['y'], gt_rd['vx'], gt_rd['vy'], gt_rd['having_truth']
        
        pc = LidarPointCloud( np.stack([x_list, y_list, np.zeros_like(x_list), np.ones_like(x_list)]) )
                    
        T_r2c = cal_trans_matrix(nusc, rd_token, cam_token1)    
        T_c2r = cal_trans_matrix(nusc, cam_token1, rd_token)
        T_c2c = cal_trans_matrix(nusc, cam_token1, cam_token2)        
        
        pc.transform(T_r2c)
        
        xi_list, yi_list, d_list, msk_in_im = proj2im(nusc, pc, cam_token1)        
        xi_list, yi_list = downsample_coord(xi_list, yi_list, downsample_scale=4, y_cutoff=33)
        
        rd_offset = np.load(join(dir_files, '%05d_offset.npy' % sample_idx))
        flow = np.load(join(dir_files, '%05d_full_flow.npy' % sample_idx))      
        u1_map, v1_map, u2_map, v2_map = flow2uv_map(flow, cx, cy, f)
        
        xi_list, yi_list = correct_coord(xi_list, yi_list, rd_offset)
        xi_list, yi_list = upsample_coord(xi_list, yi_list, downsample_scale=4, y_cutoff=33)
        
        prd_vxf = []
        prd_vyf = []
        prd_msk = []
        for xi, yi, vx, vy, d, is_in_im in zip(xi_list, yi_list, vx_list, vy_list, d_list, msk_in_im):
            
            if [xi,yi] != [-1,-1] and d_min <= d < d_max and is_in_im:           
                xp, yp = int(round(xi)), int(round(yi))        
                u1,v1,u2,v2 = u1_map[yp,xp], v1_map[yp,xp], u2_map[yp,xp], v2_map[yp,xp]                
                vx_f, vy_f, vz_f = cal_full_v_in_radar(vx, vy, d, u1, v1, u2, v2, T_c2r, T_c2c, dt)
                
                prd_vxf.append(vx_f)
                prd_vyf.append(vy_f)
                prd_msk.append(True)                            
            else:
                prd_vxf.append(0)
                prd_vyf.append(0)
                prd_msk.append(False)
                               
        vxf_list, vyf_list, gt_msk = gt_rd['vxf'], gt_rd['vyf'], gt_rd['having_truth']

        # error for the proposed
        for x, y, vx, vy, gt_vx, gt_vy, gt_valid, prd_valid in zip(x_list, y_list, prd_vxf, prd_vyf, vxf_list, vyf_list, gt_msk, prd_msk):
            if gt_valid and prd_valid:
                error = ( (vx - gt_vx)**2 + (vy - gt_vy)**2 ) ** 0.5
                
                rad_v, tan_v = decompose_v(vx, vy, x, y)
                gt_rad_v, gt_tan_v = decompose_v(gt_vx, gt_vy, x, y)
                
                error_radial = np.linalg.norm(rad_v - gt_rad_v)
                error_tan = np.linalg.norm(tan_v - gt_tan_v)
                
                error_list.append(error)
                error_radial_list.append(error_radial)
                error_tan_list.append(error_tan)
                
        # erorr for baseline
        for x, y, vx, vy, gt_vx, gt_vy, gt_valid, prd_valid in zip(x_list, y_list, vx_list, vy_list, vxf_list, vyf_list, gt_msk, prd_msk):
            if gt_valid and prd_valid:
                error = ( (vx - gt_vx)**2 + (vy - gt_vy)**2 ) ** 0.5
                
                rad_v, tan_v = decompose_v(vx, vy, x, y)
                gt_rad_v, gt_tan_v = decompose_v(gt_vx, gt_vy, x, y)
                
                error_radial = np.linalg.norm(rad_v - gt_rad_v)
                error_tan = np.linalg.norm(tan_v - gt_tan_v)
                
                error_list2.append(error)
                error_radial_list2.append(error_radial)
                error_tan_list2.append(error_tan)
                
    ave_error = np.mean(error_list)
    std_error = np.std(error_list)
    
    ave_radial_error = np.mean(error_radial_list)
    std_radial_error = np.std(error_radial_list)
    
    ave_tan_error = np.mean(error_tan_list)
    std_tan_error = np.std(error_tan_list)
    
    print('Ours')
    print('ave_error:', ave_error)
    print('std_error:', std_error)
    print('ave_radial_error:', ave_radial_error)
    print('std_radial_error:', std_radial_error)
    print('ave_tan_error:', ave_tan_error)
    print('std_tan_error:', std_tan_error)
        
    ave_error2 = np.mean(error_list2)
    std_error2 = np.std(error_list2)
    
    ave_radial_error2 = np.mean(error_radial_list2)
    std_radial_error2 = np.std(error_radial_list2)
    
    ave_tan_error2 = np.mean(error_tan_list2)
    std_tan_error2 = np.std(error_tan_list2)
    
    print('Baseline')
    print('ave_error2:', ave_error2)
    print('std_error2:', std_error2)
    print('ave_radial_error2:', ave_radial_error2)
    print('std_radial_error2:', std_radial_error2)
    print('ave_tan_error2:', ave_tan_error2)
    print('std_tan_error2:', std_tan_error2)
    
