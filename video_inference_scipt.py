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
from tqdm import tqdm
from datetime import datetime

class ViPDeepLab:
  """Sequence inference model for ViP-DeepLab.

  Frame-level ViP-DeepLab takes two consecutive frames as inputs and generates
  temporarily consistent depth-aware video panoptic predictions. Sequence-level
  ViP-DeepLab takes a sequence of images as input and propages the instance IDs
  between all 2-frame predictions made by frame-level ViP-DeepLab.

  Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
  ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
  Segmentation. CVPR, 2021.
  """

  def __init__(self, label_divisor: int):
    """Initializes a ViP-DeepLab model.

    Args:
      model_path: A string specifying the path to the exported ViP-DeepLab.
      label_divisor: An integer specifying the dataset label divisor.
    """
    self._label_divisor = label_divisor
    self._overlap_offset = label_divisor // 2
    self._combine_offset = 2 ** 32
    self.reset()

  def reset(self):
    """Resets the sequence predictions."""
    self._max_instance_id = 0
    self._stitched_panoptic = []
    self._last_panoptic = None

  def infer(self, panoptic, next_panoptic):
    """Inference for a sequence of input frames.

    Args:
      inputs: A list of tf.Tensor storing the input frames.
    """
    # Propagate instance ID from last_panoptic to next_panoptic based on ID
    # matching between panoptic and last_panoptic. panoptic and last_panoptic
    # stores panoptic predictions for the same frame but from different runs.
    next_new_mask = next_panoptic % self._label_divisor > self._overlap_offset
    if self._last_panoptic is not None:
        intersection = (
            self._last_panoptic.astype(np.int64) * self._combine_offset +
            panoptic.astype(np.int64))
        intersection_ids, intersection_counts = np.unique(
            intersection, return_counts=True)
        intersection_ids = intersection_ids[np.argsort(intersection_counts)]
        for intersection_id in intersection_ids:
            last_panoptic_id = intersection_id // self._combine_offset
            panoptic_id = intersection_id % self._combine_offset
            next_panoptic[next_panoptic == panoptic_id] = last_panoptic_id
    # Adjust the IDs for the new instances in next_panoptic.
    self._max_instance_id = max(self._max_instance_id,
                                np.max(panoptic % self._label_divisor))
    next_panoptic_cls = next_panoptic // self._label_divisor
    next_panoptic_ins = next_panoptic % self._label_divisor
    next_panoptic_ins[next_new_mask] = (
          next_panoptic_ins[next_new_mask] - self._overlap_offset
          + self._max_instance_id)
    next_panoptic = (
          next_panoptic_cls * self._label_divisor + next_panoptic_ins)
    if not self._stitched_panoptic:
        self._stitched_panoptic.append(copy.deepcopy(panoptic))
    self._stitched_panoptic.append(copy.deepcopy(next_panoptic))
    self._max_instance_id = max(self._max_instance_id,
                                  np.max(next_panoptic % self._label_divisor))
    self._last_panoptic = copy.deepcopy(next_panoptic)

  def results(self):
    """Returns the stitched panoptic predictions."""
    return self._stitched_panoptic 
      
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

def panoptic_to_three_channel(panoptic, label_divisor):
    predicted_instance_labels = panoptic % label_divisor
    predicted_semantic_labels = panoptic // label_divisor
    if np.max(predicted_semantic_labels) > 255:
        raise ValueError('Overflow: Semantic IDs greater 255 are not supported '
                        'for images of 8-bit. Please save output as numpy '
                        'arrays instead.')
    if np.max(predicted_instance_labels) > 65535:
        raise ValueError(
                        'Overflow: Instance IDs greater 65535 could not be encoded by '
                        'G and B channels. Please save output as numpy arrays instead.')
    three_channel_array = np.zeros((panoptic.shape[0], panoptic.shape[1], 3), dtype=np.uint8)
    three_channel_array[:, :, 0] = predicted_semantic_labels
    three_channel_array[:, :, 1] = predicted_instance_labels // 255
    three_channel_array[:, :, 2] = predicted_instance_labels % 255
    return three_channel_array

def get_file_name(context_name, sequence_id, camera_name, timestamp):
    return f'context:{context_name}::sequence:{sequence_id}::camera_name:{camera_name}::timestamp:{timestamp}'

def get_current_and_next_panoptic(context_predictions, file_name):
    label_divisor = waymo_constants.PANOPTIC_LABEL_DIVISOR
    file_path = os.path.join(context_predictions, file_name + '.png')
    current_panoptic = panoptic_from_three_channel(tf.io.decode_png(tf.io.read_file(file_path)), label_divisor)
    next_file_path = os.path.join(context_predictions, file_name + '_next.png')
    if os.path.exists(next_file_path):
        ''' This is the format of vip deeplab eval output'''
        next_panoptic = panoptic_from_three_channel(tf.io.decode_png(tf.io.read_file(next_file_path)), label_divisor)
    else:
        ''' This is the format of video kmax eval output'''
        current_panoptic, next_panoptic = tf.split(current_panoptic, 2, axis=0)
    return current_panoptic.numpy(), next_panoptic.numpy()

def process_single_context(context_predictions, output_dir):
    label_divisor = waymo_constants.PANOPTIC_LABEL_DIVISOR
    file_paths = tf.io.gfile.listdir(context_predictions)
    data_frame = pd.DataFrame([parse_property_values(file_path) for file_path in file_paths])
    data_frame = data_frame[~data_frame.timestamp.str.endswith('_next')]
    camera_names = list(pd.unique(data_frame['camera_name']))
    tf.io.gfile.makedirs(output_dir)
    for camera_name in camera_names:
        camera_data_frame = data_frame[data_frame['camera_name'] == camera_name].sort_values(by=['timestamp']).reset_index(drop=True)
        model = ViPDeepLab(label_divisor=label_divisor)
        for i, row in camera_data_frame.iterrows():
            if i == len(camera_data_frame) - 1:
                break
            file_name = get_file_name(row['context_name'], row['sequence_id'], row['camera_name'], row['timestamp'])
            input_array, next_input_array = get_current_and_next_panoptic(context_predictions, file_name)
            model.infer(input_array, next_input_array)

        stitched_panoptic = model.results()
        for (_, row), panoptic in zip(camera_data_frame.iterrows(), stitched_panoptic):
            file_name = get_file_name(row['context_name'], row['sequence_id'], row['camera_name'], row['timestamp'])
            three_channel_png = panoptic_to_three_channel(panoptic, label_divisor)
            new_file_path = os.path.join(output_dir, file_name + '.png')
            tf.io.write_file(new_file_path, tf.io.encode_png(three_channel_png))

def main(saved_predictions_root, output_dir_root):
    context_names = tf.io.gfile.listdir(saved_predictions_root)
    for context_name in tqdm(context_names):
        print (f"{datetime.now()}: Processing {context_name}")
        context_predictions = os.path.join(saved_predictions_root, context_name)
        output_dir = os.path.join(output_dir_root, context_name)
        process_single_context(context_predictions, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_predictions_root', type=str, required=True)
    parser.add_argument('--output_dir_root', type=str, required=True)
    args = parser.parse_args()
    main(args.saved_predictions_root, args.output_dir_root)