import tensorflow as tf
import numpy as np
from load_bird_images import load_birds

converter = tf.lite.TFLiteConverter.from_keras_model_file('bird_model.h5')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
raster_len = 256
data_dimension = 2
multi_label = 1
batch_size = 10
birds, class_numbers, class_names = load_birds(size=(200,200))
p = np.random.randint(0, len(birds), batch_size, 'int')
birds, labels = birds[p], class_numbers[p]
print('birds shape =', birds.shape)
def representative_data_gen():
    for i in range(batch_size):
        input_value = [birds[i]]
        yield [input_value]

converter.representative_dataset = representative_data_gen
tflite_model_quant = converter.convert()
tflite_model_quant_file = "bird_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)