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

  + step_root
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
  video/sequence_id: sequence ID of the frame.
  video/frame_id: ID of the frame of the video sequence.

Example to run the scipt:

   python -m deeplab2.data.build_waymo_parquet_data.py \
     --step_root=${STEP_ROOT} \
     --output_dir=${OUTPUT_DIR}
"""

import math
import os

from typing import Iterator, List, Sequence, Tuple, Optional, Union

from absl import app
from absl import flags
from absl import logging
import numpy as np
from tqdm import tqdm
from dask.distributed import Client as DDClient, LocalCluster

from PIL import Image

import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2

from deeplab2.data import data_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('step_root', None, 'STEP dataset root folder.')

flags.DEFINE_string('output_dir', None,
                    'Path to save converted TFRecord of TensorFlow examples.')
flags.DEFINE_bool(
    'use_two_frames', False, 'Flag to separate between 1 frame '
    'per TFExample or 2 consecutive frames per TFExample.')

_PANOPTIC_LABEL_FORMAT = 'png'
# _NUM_SHARDS = 10
# _IMAGE_FOLDER_NAME = 'images'
# _PANOPTIC_MAP_FOLDER_NAME = 'panoptic_maps'
# _LABEL_MAP_FORMAT = 'png'
# _INSTANCE_LABEL_DIVISOR = 1000
# _ENCODED_INSTANCE_LABEL_DIVISOR = 256
_TF_RECORD_PATTERN = '%s.tfrecord'
# _FRAME_ID_PATTERN = '%06d'

def get_context_names(step_root: str, tag: str) -> List[str]:
  paths = tf.io.gfile.glob(f'{step_root}/{tag}/*.parquet')
  return list(map(lambda path: os.path.splitext(os.path.basename(path))[0], paths))

def read(step_root: str, tag: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{step_root}/{tag}/*.parquet')
  return dd.read_parquet(paths)

# def _get_previous_frame_path(image_path: str) -> str:
#   """Gets previous frame path. If not exists, duplicate it with image_path."""
#   frame_id, frame_ext = os.path.splitext(os.path.basename(image_path))
#   folder_dir = os.path.dirname(image_path)
#   prev_frame_id = _FRAME_ID_PATTERN % (int(frame_id) - 1)
#   prev_image_path = os.path.join(folder_dir, prev_frame_id + frame_ext)
#   # If first frame, duplicates it.
#   if not tf.io.gfile.exists(prev_image_path):
#     tf.compat.v1.logging.warn(
#         'Could not find previous frame %s of frame %d, duplicate the previous '
#         'frame with the current frame.', prev_image_path, int(frame_id))
#     prev_image_path = image_path
#   return prev_image_path


def _create_panoptic_tfexample(dataframe_row: dd.Series,
                               use_two_frames: bool,
                               is_testing: bool = False) -> tf.train.Example:
  """Creates a TF example for each image.

  Args:
    image_path: Path to the image.
    panoptic_map_path: Path to the panoptic map (as an image file).
    use_two_frames: Whether to encode consecutive two frames in the Example.
    is_testing: Whether it is testing data. If so, skip adding label data.

  Returns:
    TF example proto.
  """
  cam_image = v2.CameraImageComponent.from_dict(dataframe_row)
  cam_seg_label = v2.CameraSegmentationLabelComponent.from_dict(dataframe_row)
  image_data = cam_image.image

  label_data = None
  if not is_testing:
    label_data = cam_seg_label.panoptic_label
  image_name = "How-is-this-used?"
  image_format = "jpeg"
  sequence_id = cam_seg_label.sequence_id 
  frame_id = str(cam_image.key.camera_name)
  prev_image_data = None
  prev_label_data = None
  # if use_two_frames:
  #   # Previous image.
  #   prev_image_path = _get_previous_frame_path(image_path)
  #   with tf.io.gfile.GFile(prev_image_path, 'rb') as f:
  #     prev_image_data = f.read()
  #   # Previous panoptic map.
  #   if not is_testing:
  #     prev_panoptic_map_path = _get_previous_frame_path(panoptic_map_path)
  #     prev_label_data = _decode_panoptic_map(prev_panoptic_map_path)
  return data_utils.create_video_tfexample(
      image_data,
      image_format,
      image_name,
      label_format=_PANOPTIC_LABEL_FORMAT,
      sequence_id=sequence_id,
      image_id=frame_id,
      label_data=label_data,
      prev_image_data=prev_image_data,
      prev_label_data=prev_label_data)


def _convert_dataset(step_root: str,
                     dataset_split: str,
                     output_dir: str,
                     use_two_frames: bool = False):
  """Converts the specified dataset split to TFRecord format.

  Args:
    step_root: String, Path to STEP dataset root folder.
    dataset_split: String, the dataset split (e.g., train, val).
    output_dir: String, directory to write output TFRecords to.
    use_two_frames: Whether to encode consecutive two frames in the Example.
  """
  # For val and test set, if we run with use_two_frames, we should create a
  # sorted tfrecord per sequence.
  create_tfrecord_per_sequence = ('train'
                                  not in dataset_split) and use_two_frames
  is_testing = 'test' in dataset_split

  cam_image_df = read(step_root=step_root, tag='camera_image')
  cam_seg_df = read(step_root=step_root, tag='camera_segmentation')

  image_w_seg_df = v2.merge(cam_image_df, cam_seg_df)

  # This is too costly when using the cloud storage and entire dataset
  # context_names = image_w_seg_df['key.segment_context_name'].unique().compute()
  context_names = get_context_names(step_root, 'camera_image')

  for context_name in tqdm(context_names):
    group = image_w_seg_df[image_w_seg_df['key.segment_context_name'] == context_name]
    shard_filename = _TF_RECORD_PATTERN % (context_name)
    output_filename = os.path.join(output_dir, shard_filename)
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      for i, (_, dataframe_row) in enumerate(group.iterrows()):
        example = _create_panoptic_tfexample(dataframe_row,
                                            use_two_frames, is_testing)
        tfrecord_writer.write(example.SerializeToString())


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.io.gfile.makedirs(FLAGS.output_dir)
  for dataset_split in ['train']:
  # for dataset_split in ('train', 'val', 'test'):
    logging.info('Starts to processing STEP dataset split %s.', dataset_split)
    _convert_dataset(FLAGS.step_root, dataset_split, FLAGS.output_dir,
                     FLAGS.use_two_frames)


if __name__ == '__main__':
  cluster = LocalCluster()
  dask_client = DDClient(cluster)
  print (f"Dashboard link: {dask_client.dashboard_link}")
  print ("Dask Scheduler Info: ")
  print (dask_client.scheduler_info())
  app.run(main)
