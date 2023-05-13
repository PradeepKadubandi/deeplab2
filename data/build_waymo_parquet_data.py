# coding=utf-8
# Copyright 2022 The Deeplab2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Converts Waymo Parquet data to sharded TFRecord file format with tf.train.Example protos.

The expected directory structure of the step_root parameter should be as follows:

  + root_folder
    + camera_image
      + *.parquet
    + camera_segmentation
       + *.parquet

The script should be able to process data locally or by using google cloud storage paths.

All this does is read the data from the parquet files, transform the data into the format
expected by SegmentationDecoder, and write it to a TFRecord file.

The output Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded panoptic segmentation content.
  image/segmentation/class/format: segmentation encoding format.
  image/depth/encoded: encoded depth content.
  image/depth/format: depth encoding format.
  video/sequence_id: sequence ID of the frame.
  video/frame_id: ID of the frame of the video sequence.
  next_image/encoded: encoded next-frame image content.
  next_image/segmentation/class/encoded: encoded panoptic segmentation content
    of the next frame.

Example to run the scipt:

   python -m deeplab2.data.build_waymo_parquet_data \
     --root_dir=${ROOT_DIR} \
     --output_dir=${OUTPUT_DIR} \
     --is_training_data=${IS_TRAINING_DATA} \
     --is_testing_data=${IS_TESTING_DATA}
"""

import asyncio
import math
import os
import zlib
from datetime import datetime

from typing import Iterator, List, Sequence, Tuple, Optional, Union
from collections import OrderedDict

from absl import app
from absl import flags
from absl import logging
import numpy as np
from tqdm import tqdm
from dask.distributed import Client as DDClient, LocalCluster

from PIL import Image

import tensorflow as tf
import dask.dataframe as dd
import pandas as pd
from waymo_open_dataset import v2
from waymo_open_dataset.utils import camera_segmentation_utils

from deeplab2.data import data_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', None, 'Url or path of the root directory.')

flags.DEFINE_string('output_dir', None,
                    'Path to save converted TFRecord of TensorFlow examples.')
flags.DEFINE_bool(
    'is_training_data', None, 'Flag to specify if the data is training data.')

flags.DEFINE_bool(
    'is_testing_data', None, 'Flag to specify if the data is testing data (no labels available).')

_TF_RECORD_PATTERN = '%s.tfrecord'

def get_context_names(root_dir: str, tag: str) -> List[str]:
  """Gets the path names for the camera images and panoptic maps."""
  paths = tf.io.gfile.glob(f'{root_dir}/{tag}/*.parquet')
  return list(map(lambda path: os.path.splitext(os.path.basename(path))[0], paths))

def read(root_dir: str, tag: str, context_name: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{root_dir}/{tag}/{context_name}.parquet')
  return dd.read_parquet(paths).compute()

# def _create_panoptic_tfexample(dataframe_row: dd.Series,
#                                use_two_frames: bool,
#                                is_testing: bool = False) -> tf.train.Example:
#   """Creates a TF example for each image.

#   Args:
#     image_path: Path to the image.
#     panoptic_map_path: Path to the panoptic map (as an image file).
#     use_two_frames: Whether to encode consecutive two frames in the Example.
#     is_testing: Whether it is testing data. If so, skip adding label data.

#   Returns:
#     TF example proto.
#   """
#   cam_image = v2.CameraImageComponent.from_dict(dataframe_row)
#   cam_seg_label = v2.CameraSegmentationLabelComponent.from_dict(dataframe_row)
#   image_data = cam_image.image

#   label_data = None
#   if not is_testing:
#     label_data = cam_seg_label.panoptic_label
#   image_name = "How-is-this-used?"
#   image_format = "jpeg"
#   sequence_id = cam_seg_label.sequence_id 
#   frame_id = str(cam_image.key.camera_name)
#   next_image_data = None
#   next_label_data = None
#   return data_utils.create_video_and_depth_tfexample(
#       image_data,
#       image_format,
#       image_name,
#       label_format=_PANOPTIC_LABEL_FORMAT,
#       sequence_id=sequence_id,
#       image_id=frame_id,
#       label_data=label_data,
#       next_image_data=next_image_data,
#       next_label_data=next_label_data)


def _convert_dataset(root_dir: str,
                     output_dir: str,
                     is_training_data: bool,
                     is_testing_data: bool):
  """Converts the specified dataset split to TFRecord format.

  Args:
    root_dir: String, Url of directory containing the component directories.
    output_dir: String, directory to write output TFRecords to.
    is_training_data: Boolean, whether the data is for training.
    is_testing_data: Boolean, whether the data is for testing.
  """
  context_names = get_context_names(root_dir, 'camera_image')
  print (f'{datetime.now()}: Found {len(context_names)} context names in root dir {root_dir}.')

  existing_output_paths = set(tf.io.gfile.glob(f'{output_dir}/*.tfrecord'))
  processed_contexts = set(map(lambda path: os.path.splitext(os.path.basename(path))[0], existing_output_paths))
  if len(processed_contexts) > 0:
    print (f'{datetime.now()}: Found {len(processed_contexts)} context names already in output destination: {processed_contexts}, skipping them.')

  to_process_context_names = []
  for context_name in context_names:
    if context_name not in processed_contexts:
      shard_filename = f'{context_name}.tfrecord'
      output_filename = os.path.join(output_dir, shard_filename)
      to_process_context_names.append((context_name, output_filename))

  MAX_CONTEXTS_TO_PROCESS = 800 # used for testing with a smaller number.
  to_process_context_names = to_process_context_names[:MAX_CONTEXTS_TO_PROCESS]
  print (f'{datetime.now()}: Processing {len(to_process_context_names)} context names.')

  for (context_name, output_filename) in tqdm(to_process_context_names):
    try:
      process_context_name(root_dir=root_dir, 
                          output_filename=output_filename, 
                          context_name=context_name, 
                          is_for_training=is_training_data, 
                          is_for_testing=is_testing_data)
    except Exception as e:
      print (f"{datetime.now()}: Error processing context {context_name}: {e}")

def process_context_name(root_dir, output_filename, context_name, is_for_training, is_for_testing):
    print (f"{datetime.now()}: Processing context: {context_name}")
    
    context_data = read(root_dir=root_dir, tag='camera_image', context_name=context_name)    
    if not is_for_testing:
        cam_seg_df = read(root_dir=root_dir, tag='camera_segmentation', context_name=context_name)
        context_data = v2.merge(context_data, cam_seg_df)

    print (f"context_name: {context_name}:: # of records in context_data: {len(context_data)}")
    tf_records = list()
    
    groups = context_data.groupby(["[CameraSegmentationLabelComponent].sequence_id"])
    for sequence_id, group_by_sequence in groups:
        print (f"context_name: {context_name}:: start processing sequence: {sequence_id}, # of records in sequence: {len(group_by_sequence)}")
        timestamps = pd.unique(group_by_sequence["key.frame_timestamp_micros"].sort_values())
        next_timestamp_map = dict(zip(timestamps[:-1], timestamps[1:]))
        next_timestamp_map[timestamps[-1]] = timestamps[-1] # set the last timestamp as next to itself - to duplicate the last row as next frame
        
        image_protos = OrderedDict()
        label_protos = OrderedDict()
        label_values = OrderedDict()
        
        for (_, row) in group_by_sequence.iterrows():
            cam_image = v2.CameraImageComponent.from_dict(row)
            row_key = (cam_image.key.camera_name, cam_image.key.frame_timestamp_micros)
            image_protos[row_key] = cam_image
            if not is_for_testing:
                cam_seg_label = v2.CameraSegmentationLabelComponent.from_dict(row)
                label_protos[row_key] = cam_seg_label
                
        print (f"context_name: {context_name}:: # of records in image_protos: {len(image_protos)}, # of records in label_protos: {len(label_protos)}")
        if not is_for_testing:
            def get_remapped_or_original_labels(label_protos, is_for_training):
              try:
                panoptic_labels, is_tracked_masks, num_cameras_covered, panoptic_label_divisor = camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
                    segmentation_proto_list=list(label_protos.values()),
                    remap_to_global=True,
                    remap_to_sequential=is_for_training,
                    new_panoptic_label_divisor=100000
                )
              except ValueError as e:
                # Look at this issue: https://github.com/waymo-research/waymo-open-dataset/issues/668
                # This is a workaround for the above issue.
                print (f"Error decoding panoptic labels: {e}, falling back to local labels.")
                panoptic_labels, is_tracked_masks, num_cameras_covered, panoptic_label_divisor = camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
                    segmentation_proto_list=list(label_protos.values()),
                    remap_to_global=False,
                    remap_to_sequential=False,
                    new_panoptic_label_divisor=100000
                )
              return (panoptic_labels, is_tracked_masks, num_cameras_covered, panoptic_label_divisor)

            (panoptic_labels, is_tracked_masks, num_cameras_covered, panoptic_label_divisor) = get_remapped_or_original_labels(label_protos, is_for_training)
            for index, row_key in enumerate(label_protos):
                label_values[row_key] = (panoptic_labels[index], is_tracked_masks[index], num_cameras_covered[index])
                
        for row_key, image_component in image_protos.items():
            camera_name, curr_timestamp = row_key
            image_data = image_component.image
            image_format = "jpeg"
            filename = f"context:{context_name}::sequence:{sequence_id}::camera_name:{image_component.key.camera_name}::timestamp:{image_component.key.frame_timestamp_micros}"
            image_id = filename
            
            next_timestamp = next_timestamp_map[curr_timestamp]
            next_row_key = (camera_name, next_timestamp)
            next_image_data=image_protos[next_row_key].image
            
            label_data=None
            label_format=None
            next_label_data=None
            
            if not is_for_testing:
                def encode_label(label):
                    serialized = tf.io.serialize_tensor(label.astype(np.int32))
                    compressed = zlib.compress(serialized.numpy())
                    return compressed
                curr_panoptic_label, curr_is_tracked_mask, curr_num_cameras_covered = label_values[row_key]
                next_panoptic_label, next_is_tracked_mask, next_num_cameras_covered = label_values[next_row_key]
                label_data=encode_label(curr_panoptic_label)
                label_format="zlib"
                next_label_data=encode_label(next_panoptic_label)
                
            tf_record = data_utils.create_video_and_depth_tfexample(
                image_data=image_data,
                image_format=image_format,
                filename=filename,
                label_format=label_format,
                sequence_id=sequence_id,
                image_id=image_id,
                label_data=label_data,
                next_image_data=next_image_data,
                next_label_data=next_label_data,
                depth_data=None,
                depth_format=None
            )
            tf_records.append(tf_record)
        print (f"context_name: {context_name}:: finish processing sequence: {sequence_id}, # of records collected so-far: {len(tf_records)}")

    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
        for example in tf_records:
            tfrecord_writer.write(example.SerializeToString())
    print (f"{datetime.now()}: Finished Processing context: {context_name}")

# def process_context_name(step_root, output_dir, use_two_frames, is_testing, context_name):
#     shard_filename = _TF_RECORD_PATTERN % (context_name)
#     output_filename = os.path.join(output_dir, shard_filename)
#     if (os.path.exists(output_filename)):
#       return
    
#     cam_image_df = read(step_root=step_root, tag='camera_image', context_name=context_name)
#     cam_seg_df = read(step_root=step_root, tag='camera_segmentation', context_name=context_name)

#     image_w_seg_df = v2.merge(cam_image_df, cam_seg_df)

#     with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
#       for i, (_, dataframe_row) in enumerate(image_w_seg_df.iterrows()):
#         example = _create_panoptic_tfexample(dataframe_row,
#                                             use_two_frames, is_testing)
#         tfrecord_writer.write(example.SerializeToString())


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.io.gfile.makedirs(FLAGS.output_dir)
  _convert_dataset(FLAGS.root_dir, FLAGS.output_dir,
                    FLAGS.is_training_data, FLAGS.is_testing_data)

if __name__ == '__main__':
  # cluster = LocalCluster()
  # dask_client = DDClient(cluster)
  # print (f"Dashboard link: {dask_client.dashboard_link}")
  # print ("Dask Scheduler Info: ")
  # print (dask_client.scheduler_info())
  app.run(main)
