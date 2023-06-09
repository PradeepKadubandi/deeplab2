# Meant to debug the data decoder and generator code.

from deeplab2 import common, config_pb2
from deeplab2.data import data_utils, dataset as deeplab_dataset, sample_generator
from deeplab2.trainer import distribution_utils, runner_utils
import tensorflow as tf
import orbit
import time


def measure_op(operation):
    start = time.perf_counter()
    result = operation()
    end = time.perf_counter()
    return result, end-start

def main():
    is_training = True
    sample_input_file_path = "/home/pkadubandi/data/waymo-open-dataset/v_2_0_0/training/vip_deeplab_format/15795616688853411272_1245_000_1265_000.tfrecord"
    label_format = "zlib"
    # sample_input_file_path = "/home/pkadubandi/data/waymo-open-dataset/v_2_0_0/training/image_with_seg_tfrecords/15795616688853411272_1245_000_1265_000.tfrecord"
    # label_format = "png"
    dataset_config = config_pb2.DatasetOptions(
        dataset="wod_pvps_image_panoptic_seg",
        file_pattern=[sample_input_file_path],
        batch_size=1,
        crop_size=[641, 481],
        min_resize_value=[641, 481],
        max_resize_value=[641, 481],
        thing_id_mask_annotations=True,
        max_thing_id=256,
        use_height_concat=True,
        decode_groundtruth_label=True
    )

    dataset_info = deeplab_dataset.MAP_NAME_TO_DATASET_INFO[dataset_config.dataset]
    decoder = data_utils.VideoKMaxDecoder(
        is_panoptic_dataset=True,
        is_video_dataset=dataset_info.is_video_dataset,
        decode_groundtruth_label=dataset_config.decode_groundtruth_label,
        label_format=label_format)

    augmentation_options = dataset_config.augmentations
    if augmentation_options.HasField('panoptic_copy_paste'):
        panoptic_copy_paste_options = augmentation_options.panoptic_copy_paste
    else:
        panoptic_copy_paste_options = None
    generator_kwargs = dict(
        dataset_info=dataset_info._asdict(),
        is_training=is_training,
        crop_size=dataset_config.crop_size,
        min_resize_value=dataset_config.min_resize_value,
        max_resize_value=dataset_config.max_resize_value,
        resize_factor=dataset_config.resize_factor,
        min_scale_factor=augmentation_options.min_scale_factor,
        max_scale_factor=augmentation_options.max_scale_factor,
        scale_factor_step_size=augmentation_options.scale_factor_step_size,
        autoaugment_policy_name=augmentation_options.autoaugment_policy_name,
        only_semantic_annotations=False,
        thing_id_mask_annotations=dataset_config.thing_id_mask_annotations,
        max_thing_id=dataset_config.max_thing_id,
        sigma=dataset_config.sigma,
        focus_small_instances=None,
        panoptic_copy_paste_options=panoptic_copy_paste_options)
    generator = sample_generator.PanopticSampleGenerator(**generator_kwargs)

    dataset = tf.data.TFRecordDataset([sample_input_file_path])
    max_samples = None
    iterator = tf.nest.map_structure(iter, dataset)
    dataset_iter_time = 0.0
    decoder_time = 0.0
    generator_time = 0.0
    sample_count = 0
    while True:
        try:
            example, op_time = measure_op(lambda: next(iterator))
            dataset_iter_time += op_time
        except StopIteration:
            break
        example, op_time = measure_op(lambda: decoder(example))
        decoder_time += op_time
        # print (f"Original image shape: {example['image'].shape}")
        # print (f"Image name: {example['image_name']}")
        example, op_time = measure_op(lambda: generator(example))
        generator_time += op_time

        # To observe the shapes of all key value pairs
        # shapes = {}
        # for key, value in example.items():
        #     shapes[key] = list(value.shape)
        # print (shapes)

        # Inspect the thing id mask
        instance_ids, _, instance_counts = tf.unique_with_counts(tf.reshape(example[common.GT_THING_ID_MASK_KEY], [-1]))
        print (f"Instance counts: {dict(zip(instance_ids.numpy(), instance_counts.numpy()))}")
        sample_count += 1
        if max_samples != None and sample_count >= max_samples:
            break

    print (f"Total samples: {sample_count}")
    print (f"Average Dataset iterator time: {dataset_iter_time / sample_count}")
    print (f"Average Decoder time: {decoder_time / sample_count}")
    print (f"Average Generator time: {generator_time / sample_count}")    
    
if __name__ == '__main__':
    main()