import tensorflow as tf
from absl import app
from absl import flags
import json
import cv2
import numpy as np
import os
import glob

FLAGS = flags.FLAGS

flags.DEFINE_string('calib_dir', './calib_data',
                    'directory of testing images')
flags.DEFINE_string('saved_model_dir', None,
                    'saved_model directory containg inference model')
flags.DEFINE_string('out_saved_model_dir', None,
                    'saved_model directory containg inference model')

FP16 = False

def main(argv):
    # See following link for details.
    # https://www.tensorflow.org/api_docs/python/tf/experimental/tensorrt/Converter
    if not FP16:
        params = tf.experimental.tensorrt.ConversionParams(
            precision_mode='INT8',
            # Set this to a large enough number so it can cache all the engines.
            # Currently only one INT8 engine is supported in this mode.
            maximum_cached_engines=1)
    else:
        params = tf.experimental.tensorrt.ConversionParams(
            precision_mode='FP16', maximum_cached_engines=16)
    converter = tf.experimental.tensorrt.Converter(
        input_saved_model_dir=FLAGS.saved_model_dir, conversion_params=params)
    
    # Define a generator function that yields input data, and use it to execute
    # the graph to build TRT engines.

    def my_input_fn():
        for name in glob.glob(os.path.join(FLAGS.calib_dir, '*.jpg')):
            image_org = cv2.imread(name)
            image = cv2.resize(image_org, (512, 640))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            yield tf.constant(image[None, ...])

    if not FP16:
        converter.convert(calibration_input_fn=my_input_fn)
    else:
        converter.convert()

    converter.build(input_fn=my_input_fn)  # Generate corresponding TRT engines
    converter.save(FLAGS.out_saved_model_dir)  # Generated engines will be saved.

if __name__ == '__main__':
    app.run(main)
