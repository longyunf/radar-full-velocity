import os
import numpy as np
from pyquaternion import Quaternion
from functools import reduce
from shapely.geometry import Point, MultiPoint
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix


def cal_matrix_refSensor_from_global(nusc, sensor_token):    
    sensor_data = nusc.get('sample_data', sensor_token)    
    ref_pose_rec = nusc.get('ego_pose', sensor_data['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])    
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)    
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)        
    M_ref_from_global = reduce(np.dot, [ref_from_car, car_from_global])    
    return M_ref_from_global


def cal_matrix_refSensor_to_global(nusc, sensor_token):    
    sensor_data = nusc.get('sample_data', sensor_token)       
    current_pose_rec = nusc.get('ego_pose', sensor_data['ego_pose_token'])
    global_from_car = transform_matrix(current_pose_rec['translation'],
                                       Quaternion(current_pose_rec['rotation']), inverse=False)
    current_cs_rec = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
    car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),                                        inverse=False)    
    M_ref_to_global = reduce(np.dot, [global_from_car, car_from_current])    
    return M_ref_to_global


def cal_trans_matrix(nusc, sensor1_token, sensor2_token):
    '''
    calculate transformation matrix from sensor1 to sensor2 (4 x 4)
    '''           
    M_ref_to_global = cal_matrix_refSensor_to_global(nusc, sensor1_token)    
    M_ref_from_global = cal_matrix_refSensor_from_global(nusc, sensor2_token)
    trans_matrix = reduce(np.dot, [M_ref_from_global, M_ref_to_global])   
    return trans_matrix


def flow2uv_map(flow, cx, cy, f):
    # flow: h x w x 2
    h, w = flow.shape[:2]
    x1_map, y1_map = np.meshgrid(np.arange(0, w), np.arange(0, h))
    dx_map, dy_map = flow[...,0], flow[...,1]
    x2_map, y2_map = x1_map + dx_map, y1_map + dy_map    
    u1_map, v1_map, u2_map, v2_map = (x1_map - cx)/f, (y1_map - cy)/f, (x2_map - cx)/f, (y2_map - cy)/f
    
    return u1_map, v1_map, u2_map, v2_map


def proj2im(nusc, pc_cam, cam_token, min_z = 2):            
    cam_data = nusc.get('sample_data', cam_token) 
    cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])         
    depth = pc_cam.points[2]    
    msk = pc_cam.points[2] >= min_z       
    points = view_points(pc_cam.points[:3, :], np.array(cs_rec['camera_intrinsic']), normalize=True)        
    x, y = points[0], points[1]
    msk =  reduce(np.logical_and, [x>0, x<1600, y>0, y<900, msk])        
    return x, y, depth, msk    


def loadMovingRadar(nusc, radar_token, disable_filters = True):  
    radar_sample = nusc.get('sample_data', radar_token)       
    pcl_path = os.path.join(nusc.dataroot, radar_sample['filename'])
    if disable_filters:
        RadarPointCloud.disable_filters()
    pc = RadarPointCloud.from_file(pcl_path)   

    pts = pc.points    
    dynamic_prop = pts[3] 
    msk_mv = reduce(np.logical_or, [dynamic_prop==0, dynamic_prop==2, dynamic_prop==6])  # moving mask
       
    x, y = pts[0], pts[1]
    vx, vy = pts[8], pts[9]
    
    x, y = x[msk_mv], y[msk_mv]
    vx, vy = vx[msk_mv], vy[msk_mv]
    
    return x,y,vx,vy   


class gt_box_key:  
    def __init__(self, nusc, sample_idx, thres_v = 0.4, disable_radar_filters = True):

        def judge_moving(v, thres_v): 
            if np.isnan( v[0] ):
                return False                                 
            v_L2 = (v[0] ** 2 + v[1] ** 2) ** 0.5            
            if v_L2 > thres_v:
                return True
            else:
                return False
                       
        sample = nusc.sample[sample_idx] 
        
        self.cam_token = nusc.sample[sample_idx]['data']['CAM_FRONT']
        self.radar_token = nusc.sample[sample_idx]['data']['RADAR_FRONT']
                        
        self.boxes_world = {}       # boxes in world coordinates                  
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)            
            v = nusc.box_velocity(ann['token'])            
            is_moving = judge_moving(v, thres_v)            
            if is_moving and ( 'vehicle' in ann['category_name'] ):
                obj_token = ann['instance_token']
                self.boxes_world[obj_token] = { k : ann[k] for k in ['translation', 'size', 'rotation', 'instance_token', 'category_name'] }
                self.boxes_world[obj_token]['v'] = v
                              
        self.boxes_radar = {}       # boxes in radar coordinates
        M_radar_from_global = cal_matrix_refSensor_from_global(nusc, self.radar_token)                        
        for obj_token in self.boxes_world:
            box = self.boxes_world[obj_token]            
            M_object_to_global = transform_matrix(box['translation'], Quaternion(box['rotation']), inverse=False)
            w,l,h = box['size']  
            vx, vy, vz = box['v']   # global coordinates            
            corners_l = np.array([[-l/2,l/2,l/2,-l/2],[-w/2,-w/2,w/2,w/2], [-h/2,-h/2,-h/2,-h/2], [0,0,0,0]])   
            corners_h = np.array([[-l/2,l/2,l/2,-l/2],[-w/2,-w/2,w/2,w/2], [h/2,h/2,h/2,h/2], [0,0,0,0]])         
            center = np.array([[0],[0],[0],[0]])
            v = np.array([[0,vx],[0,vy], [0,vz], [0,0]]) 
            keyPts = LidarPointCloud(np.concatenate([corners_l, corners_h, center], axis=1))           
            keyPts.transform(M_object_to_global)          
            pts = np.concatenate([keyPts.points,v],axis=1)            
            keyPts.points = pts        
            keyPts.transform(M_radar_from_global)
                        
            x_list = keyPts.points[0, 0:-3]
            y_list = keyPts.points[1, 0:-3]
            box_center = keyPts.points[0:2, -3]
            
            if box_center[0] > 1:           
                vx = keyPts.points[0,-1] - keyPts.points[0,-2]
                vy = keyPts.points[1,-1] - keyPts.points[1,-2]
                vz = keyPts.points[2,-1] - keyPts.points[2,-2]                
                polygon = MultiPoint( [(x,y) for (x,y) in zip(x_list, y_list)] ).convex_hull                                
                v =(vx, vy, vz)                               
                self.boxes_radar[obj_token] = {'v': v, 'center': box_center, 'polygon': polygon}
              
        self.boxes_im= {}    # boxes on image
        M_cam_from_global = cal_matrix_refSensor_from_global(nusc, self.cam_token)                        
        for obj_token in self.boxes_world:
            box = self.boxes_world[obj_token]            
            M_object_to_global = transform_matrix(box['translation'], Quaternion(box['rotation']), inverse=False)
            w,l,h = box['size']  
            vx, vy, vz = box['v']   # global coordinates
            corners_l = np.array([[-l/2,l/2,l/2,-l/2],[-w/2,-w/2,w/2,w/2], [-h/2,-h/2,-h/2,-h/2], [0,0,0,0]])   
            corners_h = np.array([[-l/2,l/2,l/2,-l/2],[-w/2,-w/2,w/2,w/2], [h/2,h/2,h/2,h/2], [0,0,0,0]])         
            center = np.array([[0],[0],[0],[0]])            
            v = np.array([[0,vx],[0,vy], [0,vz], [0,0]]) 
            keyPts = LidarPointCloud(np.concatenate([corners_l, corners_h, center], axis=1))                      
            keyPts.transform(M_object_to_global)
            pts = np.concatenate([keyPts.points,v],axis=1)
            keyPts.points = pts
            keyPts.transform(M_cam_from_global)
            d_box = keyPts.points[2,-3]
                       
            if d_box > 1:
                keyPts.points = keyPts.points[:, 0:-3]
                xs, ys, _, msks = proj2im(nusc, keyPts, self.cam_token, min_z = 2)             
                polygon = MultiPoint( [(x,y) for (x,y) in zip(xs,ys)] ).convex_hull                   
                vx = keyPts.points[0,-1] - keyPts.points[0,-2]
                vy = keyPts.points[1,-1] - keyPts.points[1,-2]
                vz = keyPts.points[2,-1] - keyPts.points[2,-2]                
                v =(vx, vy, vz)
                self.boxes_im[obj_token] = {'v': v, 'd_box_center': d_box, 'polygon': polygon }
                      
        obj_list = []
        vxf_list, vyf_list = [], []
        vxf_w_list, vyf_w_list = [], []  # in global coordinates
        having_truth = []
        x_list, y_list, vx_list, vy_list = loadMovingRadar(nusc, self.radar_token, disable_filters = disable_radar_filters)
        
        # filter out points outside image
        pc = LidarPointCloud( np.stack([x_list, y_list, np.zeros_like(x_list), np.ones_like(x_list)]) )    
        T_r2c = cal_trans_matrix(nusc, self.radar_token, self.cam_token)       
        pc.transform(T_r2c)        
        x_i, y_i, depth, msk = proj2im(nusc, pc, self.cam_token)
        x_i, y_i, depth = x_i[msk], y_i[msk], depth[msk]
        x_list, y_list, vx_list, vy_list = x_list[msk], y_list[msk], vx_list[msk], vy_list[msk]
        
        for x,y,vx,vy in zip(x_list, y_list, vx_list, vy_list):            
            p1 = Point(x,y)
            box_found = False
            for obj_token in self.boxes_radar:
                box = self.boxes_radar[obj_token]
                poly = box['polygon']
                if p1.distance(poly) < 0.5:
                    vxf, vyf = box['v'][:2]
                    vxf_w, vyf_w = self.boxes_world[obj_token]['v'][:2]
                    error = abs( (vxf*vx + vyf*vy)/(vx**2 + vy**2) - 1 )
                    if error < 0.2:
                        obj_list.append(obj_token)
                        vxf_list.append(vxf)
                        vyf_list.append(vyf)
                        vxf_w_list.append(vxf_w)
                        vyf_w_list.append(vyf_w)
                        having_truth.append(True)
                        box_found = True
                        break
            if box_found == False:
                obj_list.append('')
                vxf_list.append(0)
                vyf_list.append(0)
                vxf_w_list.append(0)
                vyf_w_list.append(0)
                having_truth.append(False)
        
        # (x,y) radar coordinates; (x_i, y_i, depth) image depth
        # (vx,vy): radial velocity; (vxf,vyf): GT full velocity
        self.radar = {'x': x_list, 'y': y_list, 'x_i': x_i, 'y_i': y_i, 'depth': depth, 'vx': vx_list, 'vy':vy_list, 'vxf': vxf_list, 'vyf': vyf_list, 'vxf_w': vxf_w_list, 'vyf_w': vyf_w_list, 'having_truth': having_truth, 'obj_list': obj_list}
        
    def v_label_exist(self, thres_n_gt_pts=2):
        having_truth = self.radar['having_truth']
        if len(having_truth)>0 and np.sum(having_truth) >= thres_n_gt_pts:
            return True
        else:
            return False
        
    
def cal_full_v(vx, vy, d, u1, v1, u2, v2, T_c2r, T_c2c, dt):    
    '''
    inputs:
        (vx,vy): radial velocity in radar coordinates
        d: depth in camera coordinates
        u1,v1,u2,v2: image flow 
    outputs:
        full velocity in camera coordinates 
    '''        
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
               
    return (vx_c, vy_c, vz_c)


def transform_velocity(v, T):      
    vx, vy, vz = v[0], v[1], v[2]             
    v2 = np.squeeze( np.dot(T[:3,:3], np.array([[vx], [vy], [vz]])) )   
    return v2
             
     
def cal_depthMap(rd, uv, Tc2r, Tc2c, Tc2w, dt, nb, downsample_scale, y_cutoff):
        
    x_i, y_i, depth, vx, vy, msk = rd['x_i'], rd['y_i'], rd['depth'], rd['vx'], rd['vy'], rd['having_truth']   
    vx_gt, vy_gt = rd['vxf_w'], rd['vyf_w']  
    h_im, w_im = 900, 1600
    n_nb = len(nb.xy)
    h_new = int( h_im / downsample_scale ) - y_cutoff
    w_new = int( w_im / downsample_scale ) 
      
    depth_map = np.zeros( (h_new, w_new) , dtype=float) 
    msk_map = np.zeros( (h_new, w_new) , dtype=bool)
    
    vx_gt_map = np.zeros( (h_new, w_new) , dtype=float) 
    vy_gt_map = np.zeros( (h_new, w_new) , dtype=float) 
    
    vx_nb_map = np.zeros( (h_new, w_new, n_nb) , dtype=float) 
    vy_nb_map = np.zeros( (h_new, w_new, n_nb) , dtype=float) 
    vz_nb_map = np.zeros( (h_new, w_new, n_nb) , dtype=float) 
    msk_nb_map = np.zeros( (h_new, w_new, n_nb) , dtype=bool) 
       
    x_i = (x_i + 0.5) / downsample_scale - 0.5
    y_i = (y_i + 0.5) / downsample_scale - 0.5 - y_cutoff      
    x_i = np.clip(x_i, 0, w_new - 1)
    y_i = np.clip(y_i, 0, h_new - 1)
           
    for i in range(len(x_i)):
        x_one, y_one = int(round( x_i[i] )), int(round( y_i[i] ))               
        if depth_map[y_one,x_one] == 0 or depth_map[y_one,x_one] > depth[i]:
            depth_map[y_one,x_one] = depth[i] 
            msk_map[y_one,x_one] = msk[i]
            for nb_idx, (ox,oy) in enumerate(nb.xy):
                x_new = x_one + ox
                y_new = y_one + oy
                if (0 <= x_new < w_new) and (0 <= y_new < h_new):
                    msk_nb_map[y_one, x_one, nb_idx] = True
                    u1, v1, u2, v2 = uv[y_new,x_new]                    
                    v_cam = cal_full_v(vx[i], vy[i], depth[i], u1, v1, u2, v2, Tc2r, Tc2c, dt)
                    v_world = transform_velocity(v_cam, Tc2w) 
                    vx_nb_map[y_one, x_one, nb_idx] = v_world[0]
                    vy_nb_map[y_one, x_one, nb_idx] = v_world[1]
                    vz_nb_map[y_one, x_one, nb_idx] = v_world[2] 
                    vx_gt_map[y_one, x_one] = vx_gt[i]  
                    vy_gt_map[y_one, x_one] = vy_gt[i]                            

    return depth_map, msk_map, msk_nb_map, vx_nb_map, vy_nb_map, vz_nb_map, vx_gt_map, vy_gt_map

