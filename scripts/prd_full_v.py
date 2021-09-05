"""
Predict and visualize full velocity
"""
import os
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import argparse
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

import _init_paths
from gt_velocity import gt_box_key, cal_trans_matrix, proj2im, flow2uv_map
from pt_wise_error import get_im_pair, cal_full_v_in_radar, correct_coord, upsample_coord, downsample_coord


def plot_flow(im, flow, title='flow', color='cyan', step=20):
    h, w = im.shape[:2]
    
    x1, y1 = np.meshgrid(np.arange(0,w), np.arange(0,h))
    
    dx = flow[...,0]
    dy = flow[...,1]
    
    # plt.figure()
    plt.imshow(im)
    for i in range(0, h, step):
        for j in range(0, w, step):
            plt.arrow(x1[i,j], y1[i,j], dx[i,j], dy[i,j], length_includes_head=True, width=0.2, head_width=2, color=color)
            
    plt.title(title)
    plt.show()


def pltRadarWithV(x,y,vx,vy, color='red', zorder=1):
    plt.scatter(x,y,s=5)    
    for i in range(len(x)):
        plt.arrow(x[i], y[i], vx[i], vy[i], length_includes_head=True, width=0.05, head_width=0.3, color=color, zorder=zorder)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str)  
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='dataset split')
    parser.add_argument('--sample_idx', type=int, default=17659)
    
    args = parser.parse_args()    
      
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')
     
    args.dir_nuscenes = join(args.dir_data, 'nuscenes')
    dir_files = join(args.dir_data, 'prepared_data') 
        
    sample_indices = torch.load(join(args.dir_data,'sample_split.tar'))['test_sample_indices']    
    nusc = NuScenes(args.version, dataroot = args.dir_nuscenes, verbose=False) 
        
    cam_token1 = nusc.sample[args.sample_idx]['data']['CAM_FRONT']
    im1, im2, f, cx, cy, dt, cam_token2 = get_im_pair(nusc, cam_token1)
    
    gt = gt_box_key(nusc, args.sample_idx)        
    rd_token = gt.radar_token
    
    gt_rd = gt.radar
    x_list, y_list, vx_list, vy_list, msk_gt = gt_rd['x'], gt_rd['y'], gt_rd['vx'], gt_rd['vy'], gt_rd['having_truth']
    
    pc = LidarPointCloud( np.stack([x_list, y_list, np.zeros_like(x_list), np.ones_like(x_list)]) )
                
    T_r2c = cal_trans_matrix(nusc, rd_token, cam_token1)    
    T_c2r = cal_trans_matrix(nusc, cam_token1, rd_token)
    T_c2c = cal_trans_matrix(nusc, cam_token1, cam_token2)        
    
    pc.transform(T_r2c)
    
    xi_list, yi_list, d_list, msk_in_im = proj2im(nusc, pc, cam_token1)    
    xi_list0, yi_list0 = xi_list, yi_list    
    xi_list, yi_list = downsample_coord(xi_list, yi_list, downsample_scale=4, y_cutoff=33)
    
    rd_offset = np.load(join(dir_files, '%05d_offset.npy' % args.sample_idx))
    flow = np.load(join(dir_files, '%05d_full_flow.npy' % args.sample_idx))      
    u1_map, v1_map, u2_map, v2_map = flow2uv_map(flow, cx, cy, f)
    
    xi_list, yi_list = correct_coord(xi_list, yi_list, rd_offset)
    xi_list, yi_list = upsample_coord(xi_list, yi_list, downsample_scale=4, y_cutoff=33)
    
    prd_vxf = []
    prd_vyf = []
    prd_msk = []
    for xi, yi, vx, vy, d, is_in_im in zip(xi_list, yi_list, vx_list, vy_list, d_list, msk_in_im):
               
        if [xi,yi] != [-1,-1]:        
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
    
    plt.close('all')
    ## plot raw radar depth and radar-pixel association
    plt.figure()
    plt.imshow(im1)
    for x0,y0,x1,y1 in zip(xi_list0, yi_list0, xi_list, yi_list):
        if x1 != -1:
            plt.arrow(x=x0, y=y0, dx=(x1-x0), dy=(y1-y0), length_includes_head=True, width=2, head_width=5, color='yellow')  
    plt.scatter(xi_list0, yi_list0, c=d_list, s=10, cmap='jet')
       
    plt.figure()
    plot_flow(im1, flow)    
      
    # plot predicted velocity and Doppler velocity
    plt.figure()
    for x, y, vx, vy, prd_vx, prd_vy, gt_vx, gt_vy, gt_valid, prd_valid in zip(x_list, y_list, vx_list, vy_list, prd_vxf, prd_vyf, vxf_list, vyf_list, gt_msk, prd_msk):
        if gt_valid and prd_valid:            
            pltRadarWithV([x],[y],[prd_vx],[prd_vy], 'black', zorder=5)          
        pltRadarWithV([x],[y],[vx],[vy], 'red')
    
    # plot box with GT velocity
    for obj_token in gt.boxes_radar:
        box_rd = gt.boxes_radar[obj_token]
        
        poly = box_rd['polygon']
        xy_list = list(poly.exterior.coords)
        x_temp = [xy[0] for xy in xy_list]
        y_temp = [xy[1] for xy in xy_list]
        
        plt.plot(x_temp, y_temp, color='purple', linewidth=2)
        x_ct, y_ct = box_rd['center'][:2]
        plt.arrow(x_ct, y_ct, dx=box_rd['v'][0], dy=box_rd['v'][1], length_includes_head=True, width=0.08, head_width=0.3, color='green')
        plt.axis('equal')       
    plt.show()
            
   