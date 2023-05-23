# Taken as is from Vip_Deeplab_Demo notebook in this folder. Will modify it to meet our needs.
#@title Define ViP-DeepLab Sequence Inference Class
import argparse
import copy
import os
import pandas as pd
import tensorflow as tf
import typing 
import numpy as np
from deeplab2.data import waymo_constants
from deeplab2.data import dataset
from tqdm import tqdm
from datetime import datetime
from waymo_open_dataset import v2
from waymo_open_dataset import dataset_pb2 as open_dataset
import scipy.optimize as so

dataset_info = dataset.MAP_NAME_TO_DATASET_INFO[dataset._WOD_PVPS_IMAGE_PANOPTIC_SEG]
thing_list = dataset_info.class_has_instances_list
label_divisor = dataset_info.panoptic_label_divisor
colormap_name = dataset_info.colormap
camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                              open_dataset.CameraName.FRONT_LEFT,
                              open_dataset.CameraName.FRONT,
                              open_dataset.CameraName.FRONT_RIGHT,
                              open_dataset.CameraName.SIDE_RIGHT]

def parse_property_values(file_name):
    parts = os.path.splitext(file_name)[0].split('::')
    properties = ["context_name", "sequence_id", "camera_name", "timestamp"]
    property_values = dict()
    for i, part in enumerate(parts):
        property_values[properties[i]] = part.split(':')[1]
    return property_values

def panoptic_from_three_channel(three_channel_array, label_divisor):
    three_channel_array = tf.cast(three_channel_array, tf.int32)
    instance = three_channel_array[:, :, 1] * 255 + three_channel_array[:, :, 2]
    panoptic = three_channel_array[:, :, 0] * label_divisor + instance
    return panoptic

def get_three_channel_from_sem_inst(predicted_semantic_labels, predicted_instance_labels):
    predicted_semantic_labels = np.squeeze(predicted_semantic_labels, axis=-1)
    predicted_instance_labels = np.squeeze(predicted_instance_labels, axis=-1)
    if np.max(predicted_semantic_labels) > 255:
        raise ValueError('Overflow: Semantic IDs greater 255 are not supported '
                        'for images of 8-bit. Please save output as numpy '
                        'arrays instead.')
    if np.max(predicted_instance_labels) > 65535:
        raise ValueError(
                        'Overflow: Instance IDs greater 65535 could not be encoded by '
                        'G and B channels. Please save output as numpy arrays instead.')
    three_channel_array = np.zeros((predicted_semantic_labels.shape[0], predicted_semantic_labels.shape[1], 3), dtype=np.uint8)
    three_channel_array[:, :, 0] = predicted_semantic_labels
    three_channel_array[:, :, 1] = predicted_instance_labels // 255
    three_channel_array[:, :, 2] = predicted_instance_labels % 255
    return three_channel_array

def get_file_name(context_name, sequence_id, camera_name, timestamp):
    return f'context:{context_name}::sequence:{sequence_id}::camera_name:{camera_name}::timestamp:{timestamp}'

def get_current_panoptic_three_channel(context_predictions, file_name):
    file_path = os.path.join(context_predictions, file_name + '.png')
    current_panoptic = tf.cast(tf.io.decode_png(tf.io.read_file(file_path)), tf.int32)
    return current_panoptic.numpy()

def read_parquet_df(tag: str, context_name) -> pd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  dataset_dir = '/home/pkadubandi/data/waymo-open-dataset/v_2_0_0/testing_location/'
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  return pd.read_parquet(paths)

def calculate_camera_mappings(context_name):
    disk_cache = os.path.join('/home/pkadubandi/data/waymo-open-dataset/v_2_0_0/testing_location/camera_mappings_saved', context_name)
    if os.path.exists(disk_cache):
        cam_dim_ordered = np.load(os.path.join(disk_cache, 'cam_dim_ordered.npy'))
        map_cam_to_vir_ordered = np.load(os.path.join(disk_cache, 'map_cam_to_vir_ordered.npy'))
        return cam_dim_ordered, map_cam_to_vir_ordered
    
    cam_cal_df = read_parquet_df("camera_calibration", context_name)
    # Read & shape camera extrinsics
    cam_ext = np.zeros((5,4,4))
    print(cam_ext.shape)
    for i, (_, r) in enumerate(cam_cal_df.iterrows()):
        cam_cal = v2.CameraCalibrationComponent.from_dict(r)
        #print(r["key.camera_name"])
        #print(np.reshape(cam_cal.extrinsic.transform,[4,4]))
        cam_ext[i,:,:] = np.reshape(cam_cal.extrinsic.transform,[4,4])
    print(cam_ext)

    t = np.mean(cam_ext[:,:,3],axis=0)
    vir_ext = cam_ext[0].copy()
    vir_ext[:, 3] = t
    vir_ext[0, 3] = 1.1
    print(vir_ext)
    print(cam_ext[1])

    # Read and shape camera intrinsics & dimensions
    cam_int = np.zeros((5,3,3))
    cam_dim = np.zeros((5,2))
    for i, (_, r) in enumerate(cam_cal_df.iterrows()):
        cam_cal = v2.CameraCalibrationComponent.from_dict(r)
        #print(r["key.camera_name"])
        #print(cam_cal.intrinsic)
        cam_int[i,:,:] = np.array([[cam_cal.intrinsic.f_u, 0, cam_cal.intrinsic.c_u], [0, cam_cal.intrinsic.f_v, cam_cal.intrinsic.c_v], [0, 0, 1]])
        cam_dim[i, :] = np.array([cam_cal.height, cam_cal.width])  

    print(cam_int)
    print(cam_dim)

    vir_dim = [1280, 1920*5]

    # Compute mapping from a given camera to virtual camera
    # using cam_ext, cam_int, cam_dim, vir_ext, vir_int, vir_dim
    # camera L->R name  4 2 1 3 5
    radius = 100 # meters
    R_imgcam_to_vehcam = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    #print("R: ", R_imgcam_to_vehcam)
    map_cam_to_vir = np.zeros((5, 2, *cam_dim[0].astype(int)))
    for cam_idx in range(0,5):
        print("cam_idx: ", cam_idx)
        #print(map_cam_to_vir.shape)
        T_cam_to_veh = cam_ext[cam_idx].copy()
        #print("T_cam_to_veh \n", T_cam_to_veh)
        T_veh_to_vir = np.linalg.inv(vir_ext)
        #print("T_veh_to_vir \n", T_veh_to_vir)
        Kinv = np.linalg.inv(cam_int[cam_idx])
        #print("Kinv \n", Kinv)
        for x in range(0, cam_dim[cam_idx, 1].astype(int)):
            for y in range(0, cam_dim[cam_idx, 0].astype(int)):
                Pi = np.array([x, y, 1.0])
                #print("Pi ", Pi)
                Pc = np.matmul(Kinv, Pi)
                # Radially project
                Pc = (radius/np.linalg.norm(Pc)) * Pc
                #print("Pc ", Pc)
                Pcv = np.matmul(R_imgcam_to_vehcam, Pc)
                Pcv = np.append(Pcv, 1)
                #print("Pcv ", Pcv)
                Pveh = np.matmul(T_cam_to_veh, Pcv)
                #print("Pveh ", Pveh)
                Pv = np.matmul(T_veh_to_vir, Pveh)
                #print("Pv ", Pv)
                Pvc = np.matmul(R_imgcam_to_vehcam.transpose(), Pv[:-1])
                #print("Pvc ", Pvc)
                # Pvi = np.matmul(vir_int, Pvc) # perspective projection
                # Pvi = Pvi/Pvi[2]
                # map_cam_to_vir[cam_idx,0,y,x] = np.round(Pvi[0]) # col
                # map_cam_to_vir[cam_idx,1,y,x] = np.round(Pvi[1]) # row
                phi = np.arcsin(Pvc[1]/np.linalg.norm(Pvc))
                #print("phi: ", phi)
                theta = np.arctan2(Pvc[0],Pvc[2])
                #print("theta: ", theta)
                map_cam_to_vir[cam_idx,0,y,x] = np.round((theta/(1.5*np.pi) + 0.5)*vir_dim[1]) # col
                map_cam_to_vir[cam_idx,1,y,x] = np.round((phi/(0.25*np.pi) + 0.5)*vir_dim[0]) # row

    map_cam_to_vir_ordered = np.zeros(map_cam_to_vir.shape)
    cam_dim_ordered = np.zeros(cam_dim.shape)
    for idx, cam in enumerate(camera_left_to_right_order):
        map_cam_to_vir_ordered[idx] = map_cam_to_vir[cam-1]
        cam_dim_ordered[idx] = cam_dim[cam-1]
    print(cam_dim)
    print(cam_dim_ordered)

    os.makedirs(disk_cache, exist_ok=True)
    np.save(os.path.join(disk_cache, 'cam_dim_ordered.npy'), cam_dim_ordered)
    np.save(os.path.join(disk_cache, 'map_cam_to_vir_ordered.npy'), map_cam_to_vir_ordered)
    return cam_dim_ordered, map_cam_to_vir_ordered

def get_pano_linear_index_for_instance_label(cam, frm, inst_lbl, 
                                             instance_labels_multiframe,
                                             map_cam_to_vir_ordered):
    # get bool array of instance inst_lbl
    idx = instance_labels_multiframe[frm][cam]==inst_lbl
    if idx.shape[0] < 1280:
        idx = np.pad(idx[:,:,0],((0,1280-idx.shape[0]),(0,0)))
    else:
        idx = idx[:,:,0]
    #print(idx.shape) 
    # find rows and cols of the instance in pano & create a unique scalar index
    cols = map_cam_to_vir_ordered[cam,0,idx[:,:]]
    #print(cols)
    rows = map_cam_to_vir_ordered[cam,1,idx[:,:]]
    #print(rows)
    #print(cols.shape == rows.shape)
    return rows*1920 + cols

def compute_iou(rc1, rc2):
    intersection = np.intersect1d(rc1,rc2)
    union = np.union1d(rc1,rc2)
    iou = len(intersection)/len(union)
    return iou

def get_instance_pair_ious_for_2_cams(frm, cam1, cam2, thing_sem_lbls,
                                     instance_labels_multiframe, semantic_labels_multiframe,
                                     map_cam_to_vir_ordered):
    pairs_iou = []
    # Cams are ordered l->R
    for sem_lbl in thing_sem_lbls:
        inst_lbls1 = np.unique(instance_labels_multiframe[frm][cam1][semantic_labels_multiframe[frm][cam1]==sem_lbl])
        rcidx1 = []
        for inst_lbl in inst_lbls1:
            #print("cam1 sem_lbl, inst_lbl", sem_lbl, inst_lbl)
            rc = get_pano_linear_index_for_instance_label(cam1, frm, inst_lbl, 
                                                          instance_labels_multiframe, map_cam_to_vir_ordered)
            if len(rc) > 0:
                rcidx1.append((inst_lbl, rc))
    
        inst_lbls2 = np.unique(instance_labels_multiframe[frm][cam2][semantic_labels_multiframe[frm][cam2]==sem_lbl])
            
        rcidx2 = []
        for inst_lbl in inst_lbls2:
            #print("cam2 sem_lbl, inst_lbl", sem_lbl, inst_lbl)
            rc = get_pano_linear_index_for_instance_label(cam2, frm, inst_lbl, 
                                                          instance_labels_multiframe, map_cam_to_vir_ordered)
            if len(rc) > 0:
                rcidx2.append((inst_lbl, rc))

        # iterate over rcidx1 & rcidx2 and keep all pairs with iou > 0
        for idx1 in rcidx1:
            for idx2 in rcidx2:
                iou = compute_iou(idx1[1], idx2[1])
                if iou > 0.001:
                  pairs_iou.append([sem_lbl, idx1[0], idx2[0], iou]) # semantic label, cam 1 instance lbl, cam 2 instance lbl, iou
    
    return np.array(pairs_iou)

def find_instance_matches_from_pair_ious(pair_ious):
    # given pairwise ious between instance labels in left and right cameras
    # find best matching left and right instances
    matches = []
    sem_lbls = np.unique(pair_ious[:,0])
    for sem_lbl in sem_lbls:
        pair_sem_ious = pair_ious[pair_ious[:,0] == sem_lbl]
        # print(pair_sem_ious)
        vertex1s = np.unique(pair_sem_ious[:,1])
        vertex2s = np.unique(pair_sem_ious[:,2])
        vertex1_to_idx = dict(zip(vertex1s, range(0,len(vertex1s))))
        vertex2_to_idx = dict(zip(vertex2s, range(0,len(vertex2s))))
        idx_to_vertex1 = dict(zip(range(0,len(vertex1s)), vertex1s))
        idx_to_vertex2 = dict(zip(range(0,len(vertex2s)), vertex2s))
        #print(idx_to_vertex1)
        #print(idx_to_vertex2)
        weight = np.zeros((len(vertex1s), len(vertex2s)))
        for p in pair_sem_ious:
            weight[vertex1_to_idx[p[1]], vertex2_to_idx[p[2]]] = p[3]
        #print(weight)
        row_ind, col_ind = so.linear_sum_assignment(weight, True)
        for r, c in zip(row_ind, col_ind):
            match = [sem_lbl, idx_to_vertex1[r], idx_to_vertex2[c]]
            matches.append(match)
    return matches

def get_history_len_for_label(match_frm, cam, lbl, instance_labels_multiframe):
    num_frames = len(instance_labels_multiframe)
    max_history_len = 10
    history_len = 0
    for frm in range(match_frm-1, match_frm-max_history_len, -1):
        if frm < 0:
            break
        if np.any(instance_labels_multiframe[frm][cam]==lbl):
            history_len += 1
    return history_len

def get_replacements_from_matches_for_camera_pair(matches, match_frm, cam1, cam2, instance_labels_multiframe):
    # matches are [sem_label inst_label_cam1 inst_label_cam2]
    cam1_replacement = []
    cam2_replacement = []
    for match in matches:
        lbl_cam1 = match[1]
        lbl_cam2 = match[2]
        # check which one of them has a longer history
        hist_len1 = get_history_len_for_label(match_frm, cam1, lbl_cam1, instance_labels_multiframe)
        hist_len2 = get_history_len_for_label(match_frm, cam2, lbl_cam2, instance_labels_multiframe)
        if (hist_len1 >= hist_len2):
            cam2_replacement.append([lbl_cam2, lbl_cam1])
        else:
            cam1_replacement.append([lbl_cam1, lbl_cam2])
    
    return cam1_replacement, cam2_replacement

def propagate_instance_labels_for_cam(cam, start_frm, end_frm, inst_lbls_replacement, instance_labels_multiframe):
    for frm in range(start_frm, end_frm):
        for orig, new in inst_lbls_replacement:
            instance_labels_multiframe[frm][cam][instance_labels_multiframe[frm][cam]==orig] = new
    
    return instance_labels_multiframe

def process_single_context(context_name, context_predictions, output_dir):
    cam_dim_ordered, map_cam_to_vir_ordered = calculate_camera_mappings(context_name)
    tf.io.gfile.makedirs(output_dir)
    label_divisor = waymo_constants.PANOPTIC_LABEL_DIVISOR
    file_paths = tf.io.gfile.listdir(context_predictions)
    data_frame = pd.DataFrame([parse_property_values(file_path) for file_path in file_paths])
    data_frame = data_frame[~data_frame.timestamp.str.endswith('_next')]
    all_timestamps = pd.unique(data_frame['timestamp'].sort_values())
    semantic_labels_multiframe = []
    instance_labels_multiframe = []
    for timestamp in all_timestamps:
        multi_cam_frames = data_frame[data_frame['timestamp'] == timestamp]
        semantic_labels = []
        instance_labels = []
        for camera_name in camera_left_to_right_order:
            row = multi_cam_frames[multi_cam_frames['camera_name'] == str(camera_name)].iloc[0]
            file_name = get_file_name(row['context_name'], row['sequence_id'], row['camera_name'], row['timestamp'])
            three_channel_panoptic = get_current_panoptic_three_channel(context_predictions, file_name)
            semantic_labels.append(np.expand_dims(three_channel_panoptic[:, :, 0], axis=-1))
            instance_labels.append(np.expand_dims(three_channel_panoptic[:, :, 1] * 255 + three_channel_panoptic[:, :, 2], axis=-1))
        semantic_labels_multiframe.append(semantic_labels)
        instance_labels_multiframe.append(instance_labels)

    n_frames = len(semantic_labels_multiframe)
    for frame_idx in range(n_frames):
        for (cam1, cam2) in zip(camera_left_to_right_order[:-1], camera_left_to_right_order[1:]):
            cam1 -= 1
            cam2 -= 1
            p_iou = get_instance_pair_ious_for_2_cams(frame_idx, cam1, cam2, thing_list,
                            instance_labels_multiframe, semantic_labels_multiframe,
                            map_cam_to_vir_ordered)
            if len(p_iou) == 0:
                # print ("No matches found for frame %d, cam1 %s, cam2 %s" % (frame_idx, cam1, cam2))
                continue
            matches = find_instance_matches_from_pair_ious(p_iou)
            r1, r2 = get_replacements_from_matches_for_camera_pair(matches, frame_idx, cam1, cam2, instance_labels_multiframe)
            instance_labels_multiframe = propagate_instance_labels_for_cam(cam1, frame_idx, n_frames, r1, instance_labels_multiframe)
            instance_labels_multiframe = propagate_instance_labels_for_cam(cam2, frame_idx, n_frames, r2, instance_labels_multiframe)
    
    for frame_idx in range(n_frames):
        timestamp = all_timestamps[frame_idx]
        for camera_idx, camera_name in enumerate(camera_left_to_right_order):
            file_name = get_file_name(context_name, context_name, camera_name, timestamp)
            semantic_labels = semantic_labels_multiframe[frame_idx][camera_idx]
            instance_labels = instance_labels_multiframe[frame_idx][camera_idx]
            three_channel_png = get_three_channel_from_sem_inst(semantic_labels, instance_labels)
            new_file_path = os.path.join(output_dir, file_name + '.png')
            tf.io.write_file(new_file_path, tf.io.encode_png(three_channel_png))
            

def main(saved_predictions_root, output_dir_root):
    context_names = set(tf.io.gfile.listdir(saved_predictions_root))
    processed_context_names = set(tf.io.gfile.listdir(output_dir_root)) # to resume run that was killed for some reason.
    context_names = context_names.difference(processed_context_names)
    # context_names = context_names[:1]
    # context_names = ["13787943721654585343_1220_000_1240_000"]
    for context_name in tqdm(context_names):
        print (f"{datetime.now()}: Processing {context_name}")
        context_predictions = os.path.join(saved_predictions_root, context_name)
        output_dir = os.path.join(output_dir_root, context_name)
        try:
            process_single_context(context_name, context_predictions, output_dir)
        except Exception as e:
            print (f"Error processing {context_name}: {e}, continuing to next context.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_predictions_root', type=str, required=True)
    parser.add_argument('--output_dir_root', type=str, required=True)
    args = parser.parse_args()
    main(args.saved_predictions_root, args.output_dir_root)