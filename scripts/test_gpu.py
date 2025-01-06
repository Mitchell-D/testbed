
import tensorflow as tf
print(tf.config.list_physical_devices())
print("GPUs: ", len(tf.config.list_physical_devices('GPU')))
