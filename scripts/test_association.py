import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from os.path import join
import sys
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn

import _init_paths
from pyramidNet import PyramidCNN
from data_loader_v import init_data_loader
from nb_utils import sparse_neighbor_connection


def load_weights(args, model):
    f_checkpoint = join(args.dir_result, 'checkpoint.tar')       
    if os.path.isfile(f_checkpoint):
        print('load best model')        
        model.load_state_dict(torch.load(f_checkpoint)['state_dict_best'])
    else:
        sys.exit('No model found')
 
    
def init_env():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    cudnn.benchmark = True if use_cuda else False
    return device


def plt_depth_on_im(depth_map, im, title = '',  ptsSize = 1):
    
    h,w = im.shape[0:2]    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    msk = depth_map > 0
    
    plt.figure()
    plt.imshow(im) 
    plt.scatter(x_map[msk], y_map[msk], c=depth_map[msk], s=ptsSize, cmap='jet')
    plt.title(title, fontsize=20)
    plt.colorbar()
    plt.axis('off')


def cal_radar_position_offset(prd_aff, d_radar, nb, thres_aff=0.5):
    '''
    inputs:
        prd_aff: n_nb x h x w
        d_radar: h x w
    outputs:
        offset: numpy (h,w,2); at each pixel (dx,dy); no association (dx,dy)=(-1000,-1000)  
    '''      
    h,w = d_radar.shape    
    offset = -1000 * np.ones((h,w,2))    
    xy_list = nb.xy

    max_aff = np.max(prd_aff, axis=0)
    idx_max = np.argmax(prd_aff, axis=0)
    
    msk_radar = d_radar > 0
    max_aff = max_aff * msk_radar
            
    for i in range(h):
        for j in range(w):
            if d_radar[i,j] > 0 and max_aff[i,j] > thres_aff:
                offset[i,j,:] = xy_list[idx_max[i,j]]
                                
    return offset, max_aff


def prd_one_sample(model, nb, test_loader, device, sample_idx = 1, thres_aff = 0.3):
    
    def plt_association_on_im(depth_map, im, pos_offset, title = '',  ptsSize = 1):
        h,w = im.shape[0:2]    
        x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
        msk = depth_map > 0
        
        plt.figure()
        plt.imshow(im) 
         
        for i in range(h):
            for j in range(w):
                dx, dy = pos_offset[i,j,:]
                if [dx,dy] != [-1000,-1000]:
                    plt.arrow(j, i, dx, dy, length_includes_head=True, width=0.1, head_width=0.2, color='yellow')
                    
        plt.scatter(x_map[msk], y_map[msk], c=depth_map[msk], s=ptsSize, cmap='jet')
        plt.title(title, fontsize=20)
        plt.colorbar()
        plt.axis('off')
        plt.show()
        
    with torch.no_grad():
        for ct, sample in enumerate(test_loader):
            s_idx = sample['sample_idx']
            if s_idx == sample_idx:
                data_in = sample['data_in'].to(device)                              
                prd = torch.sigmoid( model(data_in)[0] )                 
                d_radar_tensor = data_in[:,[7],...]                
                im = data_in[0][0:3].permute(1,2,0).to('cpu').numpy()
                d_radar = d_radar_tensor[0][0].to('cpu').numpy()               
                prd = prd[0].cpu().numpy()                
                break
         
    pos_offset, max_aff = cal_radar_position_offset(prd, d_radar, nb, thres_aff)     
    plt.close('all')      
    plt_association_on_im(d_radar, im, pos_offset, title = 'Association',  ptsSize = 30) 


def gen_offset_map(model, test_loader, device, args):    
    test_indices = []  
    with torch.no_grad():
        for ct, sample in enumerate(tqdm(test_loader)): 
            data_in, sample_idx = sample['data_in'].to(device), sample['sample_idx'][0].item()             
            test_indices.append(sample_idx)           
            prd = torch.sigmoid( model(data_in)[0] )         
            d_radar = data_in[0][7].to('cpu').numpy()      
            prd = prd[0].cpu().numpy()
            pos_offset, _ = cal_radar_position_offset(prd, d_radar, args.nb, thres_aff=0.3)  
            np.save(join(args.output_folder, '%05d_offset.npy' % sample_idx), pos_offset)
 
    
def main(args):   
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')
     
    if not args.dir_result:
        args.dir_result = join(args.dir_data, 'train_result', '%d_%d_%d' % (args.left_right, args.top, args.bottom))
    args.path_data_file = join(args.dir_data, 'prepared_data.h5')    
    args.output_folder = join(args.dir_data, 'prepared_data')
    args.dir_nuscenes = join(args.dir_data, 'nuscenes')
        
    args.nb = sparse_neighbor_connection(*(args.left_right, args.left_right, args.top, args.bottom, args.skip))
    args.outChannels = len(args.nb.xy)
                   
    device = init_env()
        
    test_loader = init_data_loader(args, 'test')
    
    model = PyramidCNN(args.nLevels, args.nPred, args.nPerBlock, 
                        args.nChannels, args.inChannels, args.outChannels, 
                        args.doRes, args.doBN, doELU=False, 
                        predPix=False, predBoxes=False).to(device)
    
    load_weights(args, model)       
    model.eval()
       
    if args.gen_offset:
        gen_offset_map(model, test_loader, device, args)
    else:
        sample_idx = 17659 
        thres_aff = 0.3
        prd_one_sample(model, args.nb, test_loader, device, sample_idx, thres_aff)
      
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--dir_result', type=str)    
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='dataset split')

    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--no_data_shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
        
    parser.add_argument('--nLevels', type=int, default=5)
    parser.add_argument('--nPred', type=int, default=1)
    parser.add_argument('--nPerBlock', type=int, default=1)
    parser.add_argument('--nChannels', type=int, default=64)   
    parser.add_argument('--inChannels', type=int, default=8)
    parser.add_argument('--doRes', type=bool, default=True)
    parser.add_argument('--doBN', type=bool, default=True) 
    
    parser.add_argument('--left_right', type=int, default=4)
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--bottom', type=int, default=4)
    parser.add_argument('--skip', type=int, default=1)
    
    parser.add_argument('--gen_offset', action='store_true', default=False, help='generate predicted offsets')
   
    args = parser.parse_args()
    
    main(args)
    
   