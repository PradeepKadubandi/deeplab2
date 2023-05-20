# Taken as is from Vip_Deeplab_Demo notebook in this folder. Will modify it to meet our needs.
#@title Define ViP-DeepLab Sequence Inference Class
import copy
import tensorflow as tf
import typing 
import numpy as np

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

  def __init__(self, model_path: str, label_divisor: int):
    """Initializes a ViP-DeepLab model.

    Args:
      model_path: A string specifying the path to the exported ViP-DeepLab.
      label_divisor: An integer specifying the dataset label divisor.
    """
    self._model = tf.saved_model.load(model_path)
    self._label_divisor = label_divisor
    self._overlap_offset = label_divisor // 2
    self._combine_offset = 2 ** 32
    self.reset()

  def reset(self):
    """Resets the sequence predictions."""
    self._max_instance_id = 0
    self._depth_preds = []
    self._stitched_panoptic = []
    self._last_panoptic = None

  def _infer(self, input_array, next_input_array):
    """Inference for two consecutive input frames."""
    input_array = np.concatenate((input_array, next_input_array), axis=-1)
    output = self._model(input_array)
    depth = output['depth_pred'].numpy()
    panoptic = output['panoptic_pred'].numpy()
    next_panoptic = output['next_panoptic_pred'].numpy()
    return depth, panoptic, next_panoptic

  def infer(self, inputs: typing.List[tf.Tensor]):
    """Inference for a sequence of input frames.

    Args:
      inputs: A list of tf.Tensor storing the input frames.
    """
    self.reset()
    for input_idx in range(len(inputs) - 1):
      depth, panoptic, next_panoptic = self._infer(inputs[input_idx],
                                                   inputs[input_idx + 1])
      self._depth_preds.append(copy.deepcopy(depth))
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
    """Returns the sequence inference results."""
    return self._depth_preds, self._stitched_panoptic