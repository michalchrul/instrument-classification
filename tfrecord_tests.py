import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an
    # EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# The number of observations in the dataset.
n_observations = 1
# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)
# Float feature
feature1 = np.random.randn(n_observations)
# String feature
strings = np.array([b'cat', b'dog'])
feature2 = np.random.choice(strings, n_observations)
# Non-scalar Float feature, 2x2 matrices sampled from a standard normal
# distribution
feature3 = np.random.randn(n_observations, 2, 2)
dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2,
                                              feature3))


def create_example(feature0, feature1, feature2, feature3):
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _float_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _bytes_feature(feature3),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(
        feature=feature))

    return example_proto

for feature0, feature1, feature2, feature3 in dataset.take(1):
    example_proto = create_example(feature0,
                                 feature1,
                                 feature2,
                                 tf.io.serialize_tensor(feature3))
    #print(example_proto)

filename = 'saved_tfrecords/recording_1.tfrecord'

def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.
  # Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

