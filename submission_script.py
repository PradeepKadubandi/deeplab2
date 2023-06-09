from collections import defaultdict
import os
from typing import List, Optional, Sequence
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from deeplab2.data import waymo_constants
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import camera_segmentation_metrics_pb2 as metrics_pb2
from waymo_open_dataset.protos import camera_segmentation_submission_pb2 as submission_pb2
from waymo_open_dataset.utils import camera_segmentation_utils

'''
Replace all the global variables as appropriate.
And run:
    python -m deeplab2.submission_script

Assumes that the predictions are stored as 'three_channel_png' files in the following format:
    predictions_root/context_name/file_name.png
    where file_name is of the format:
        context:{context_name}::sequence:{sequence_id}::camera_name:{camera_name}::timestamp:{timestamp}

This will write the submission protos. Needs to run thet tar command at the end of tutorial notebook to generate a file that can be submitted.
    tar czvf /tmp/camera_segmentation_challenge/submit_testing.tar.gz -C $SAVE_FOLDER .

Finally upload the .tar.gz file to the submission website!!

'''
# Global variables.
panoptic_label_divisor = waymo_constants.PANOPTIC_LABEL_DIVISOR
camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                              open_dataset.CameraName.FRONT_LEFT,
                              open_dataset.CameraName.FRONT,
                              open_dataset.CameraName.FRONT_RIGHT,
                              open_dataset.CameraName.SIDE_RIGHT]
wod_source_folder = "/home/pkadubandi/GH/waymo-research/waymo-open-dataset"
predictions_root = "/home/pkadubandi/GH/PradeepKadubandi/waymo-challenge/saved_experiments/lamda-vm-1/pk-video-kmax-expt-6-convnextL-quarter-crop-4-batch/submission_test_set_eval_at_half_crop/video_stiched_panoptic_almost_final"

submission_type = 'testing'
TEST_SET_SOURCE = os.path.join(wod_source_folder, 'tutorial/2d_pvps_test_frames.txt') #@param {type: "string"} 

SAVE_FOLDER = os.path.join(wod_source_folder, 'submissions', submission_type) #@param {type: "string"}

submission_account_name = 'pkadubandi@gmail.com'
submission_method_name = 'Clip KMax with Video Stitching'
submission_description = 'Train Clip KMax model (Video KMax without HiLa MB) at batch size 4 and quarter crop size of clip (due to limited training compute) and do video stitching only (no panoramic stitching)'
# end of global variables

def _make_submission_proto() -> submission_pb2.CameraSegmentationSubmission:
    """Makes a submission proto to store predictions for one shard."""
    submission = submission_pb2.CameraSegmentationSubmission()
    submission.account_name = submission_account_name
    submission.unique_method_name = submission_method_name
    submission.authors.extend(['Aniket Murarka', 'Pradeep Kadubandi'])
    submission.affiliation = 'TeamAniketPradeep'
    submission.description = submission_description
    submission.method_link = 'http://example.com/'
    submission.frame_dt = 1
    submission.runtime_ms = 1000
    return submission

def get_file_name(context_name, sequence_id, camera_name, timestamp):
    return f'context:{context_name}::sequence:{sequence_id}::camera_name:{camera_name}::timestamp:{timestamp}'

def panoptic_from_three_channel(three_channel_array):
    three_channel_array = tf.cast(three_channel_array, tf.int32)
    instance = three_channel_array[:, :, 1] * 255 + three_channel_array[:, :, 2]
    panoptic = three_channel_array[:, :, 0] * panoptic_label_divisor + instance
    panoptic = tf.expand_dims(panoptic, axis=-1)
    return panoptic

def load_panoptic_prediction(context_name, camera_name, timestamp):
    file_name = get_file_name(context_name=context_name, sequence_id=context_name, camera_name=camera_name, timestamp=timestamp)
    three_channel_png = tf.io.decode_png(tf.io.read_file(os.path.join(predictions_root, context_name, file_name + '.png')))
    return panoptic_from_three_channel(three_channel_png).numpy()
    
def _load_predictions_for_one_test_shard(
    submission: submission_pb2.CameraSegmentationSubmission,
    context_name: str,
    timestamp_micros: Sequence[int],
) -> None:
  """Iterate over all test frames in one sequence and generate predictions."""
  print(f'Processing test sequence with context {context_name}...')
  for timestamp in timestamp_micros:
    for camera_name in camera_left_to_right_order:
        panoptic_pred = load_panoptic_prediction(context_name, camera_name, timestamp)
        label_sequence_id = context_name
        seg_proto = camera_segmentation_utils.save_panoptic_label_to_proto(
            panoptic_pred,
            panoptic_label_divisor,
            label_sequence_id)
        seg_frame = metrics_pb2.CameraSegmentationFrame(
            camera_segmentation_label=seg_proto,
            context_name=context_name,
            frame_timestamp_micros=timestamp,
            camera_name=camera_name
        )
        submission.predicted_segmentation_labels.frames.extend([seg_frame])

def _save_submission_to_file(
    submission: submission_pb2.CameraSegmentationSubmission,
    filename: str,
    save_folder: str = SAVE_FOLDER,
) -> None:
  """Save predictions for one sequence as a binary protobuf."""
  os.makedirs(save_folder, exist_ok=True)
  basename = os.path.basename(filename)
  if '.tfrecord' not in basename:
    raise ValueError('Cannot determine file path for saving submission.')
  submission_basename = basename.replace('_with_camera_labels.tfrecord',
                                         '_camera_segmentation_submission.binproto')
  submission_file_path = os.path.join(save_folder, submission_basename)
  print(f'Saving predictions to {submission_file_path}...\n')
  f = open(submission_file_path, 'wb')
  f.write(submission.SerializeToString())
  f.close()

def main():
    if tf.io.gfile.exists(SAVE_FOLDER):
        raise ValueError('Save folder already exists. Please delete it first.')
    
    context_name_timestamp_tuples = [x.rstrip().split(',') for x in (
        tf.io.gfile.GFile(TEST_SET_SOURCE, 'r').readlines())]

    context_names_dict = defaultdict(list)
    for context_name, timestamp_micros in context_name_timestamp_tuples:
        context_names_dict[context_name].append(int(timestamp_micros))

    # We save each test sequence in an indepedent submission file.
    for context_name in tqdm(context_names_dict):
        test_filename = f'segment-{context_name}_with_camera_labels.tfrecord'
        submission = _make_submission_proto()
        print('Submission proto size: ', len(submission.SerializeToString()))
        # We only include frames with timestamps requested from the .txt file in 
        # the submission.
        _load_predictions_for_one_test_shard(
            submission, 
            context_name,
            timestamp_micros=context_names_dict[context_name])
        print('Submission proto size: ', len(submission.SerializeToString()))
        _save_submission_to_file(submission, test_filename, SAVE_FOLDER)

if __name__ == '__main__':
    main()