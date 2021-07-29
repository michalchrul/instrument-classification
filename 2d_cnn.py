from config import *
import os

import tensorflow as tf
from librosa.core.time_frequency import mel_frequencies
from tensorflow.io import FixedLenFeature, parse_single_example
from tensorflow_datasets.core.utils import tqdm

import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import Sequential, load_model
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


print(tf.__version__)
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
sess = tf.compat.v1.Session(config=config)
#tf.compat.v1.disable_eager_execution()


#current datasets paths
train_path = [str(train_tf_prep_path), str(train_tf_prep_aug_path)]
valid_path = [str(valid_tf_prep_path), str(valid_tf_prep_aug_path)]
test_path = [str(test_tf_prep_path), str(test_tf_prep_aug_path)]
print(train_path)
print(valid_path)
print(test_path)

"""### Prepare spectrogramming and model parameters. """

def _normalize_tensorflow(S, hparams):
    return tf.clip_by_value((S - hparams.min_level_db) /
                            -hparams.min_level_db, 0, 1)

def _tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def _amp_to_db_tensorflow(x):
    return 20 * _tf_log10(tf.clip_by_value(tf.abs(x), 1e-5, 1e100))


def _stft_tensorflow(signals, hparams):
    return tf.signal.stft(
        signals,
        hparams.win_length,
        hparams.hop_length,
        hparams.n_fft,
        pad_end=True,
        window_fn=tf.signal.hann_window,
    )


def spectrogram_tensorflow(y, hparams):
    D = _stft_tensorflow(y, hparams)
    S = _amp_to_db_tensorflow(tf.abs(D)) - hparams.ref_level_db
    return _normalize_tensorflow(S, hparams)

class HParams(object):
    """ Hparams was removed from tf 2.0alpha so this is a placeholder
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hparams = HParams(
    # network
    batch_size = 256,
    num_epochs = 50,
    num_classes = 11,
    # spectrogramming
    sample_rate = 16000,
    create_spectrogram = True,
    win_length = 1024,
    n_fft = 1024, #tried 324 and it gave NaNs in loss
    hop_length= 400,
    ref_level_db = 50,
    min_level_db = -100,
    # mel scaling
    num_mel_bins = 128, #tried 100
    mel_lower_edge_hertz = 0,
    mel_upper_edge_hertz = 8000,
    # inversion
    power = 1.5, # for spectral inversion
    griffin_lim_iters = 50,
    pad=True,
    #
)


"""### Create the dataset class. """

class NSynthDataset(object):
    @tf.autograph.experimental.do_not_convert
    def __init__(
            self,
            tf_records,
            hparams,
            is_training=True,
            prefetch=1000,  # how many spectrograms to prefetch
            num_parallel_calls=10,  # how many threads should be preparing data
            n_samples=305979,  # how many items are in the dataset
            train_size=289205,
            valid_size=12678,
            test_size=4096,
            shuffle_buffer=hparams.batch_size * 50,
    ):
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.is_training = is_training
        self.nsamples = n_samples
        self.hparams = hparams
        self.prefetch = prefetch
        self.shuffle_buffer = shuffle_buffer
        # prepare for mel scaling
        if self.hparams.create_spectrogram:
            self.mel_matrix = self._make_mel_matrix()
        # create dataset of tfrecords
        self.raw_dataset = tf.data.TFRecordDataset(tf_records)
        # prepare dataset iterations
        self.dataset = self.raw_dataset.map(
            lambda x: self._parse_function(x),
            num_parallel_calls=num_parallel_calls
        )
        self.dataset = self.dataset.filter(self.predicate)
        # prepare the dataset
        self.prepare_dataset()

    def prepare_dataset(self):
        if self.is_training:
            self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.shuffle(self.shuffle_buffer)
        self.dataset = self.dataset.prefetch(self.prefetch)
        self.dataset = self.dataset.batch(hparams.batch_size)

    def _make_mel_matrix(self):
        # create mel matrix
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.hparams.num_mel_bins,
            num_spectrogram_bins=int(self.hparams.n_fft / 2) + 1,
            sample_rate=self.hparams.sample_rate,
            lower_edge_hertz=self.hparams.mel_lower_edge_hertz,
            upper_edge_hertz=self.hparams.mel_upper_edge_hertz,
            dtype=tf.dtypes.float32,
            name=None,
        )
        # gets the center frequencies of mel bands
        mel_f = mel_frequencies(
            n_mels=hparams.num_mel_bins + 2,
            fmin=hparams.mel_lower_edge_hertz,
            fmax=hparams.mel_upper_edge_hertz,
        )
        # Slaney-style mel is scaled to be approx constant energy
        # per channel (from librosa)
        enorm = tf.dtypes.cast(
            tf.expand_dims(tf.constant(2.0 /
                                       (mel_f[2: hparams.num_mel_bins + 2]
                                        - mel_f[:hparams.num_mel_bins])), 0),
                                        tf.float32,
        )
        # normalize matrix
        mel_matrix = tf.multiply(mel_matrix, enorm)
        mel_matrix = tf.divide(mel_matrix, tf.reduce_sum(mel_matrix, axis=0))

        return mel_matrix

    def print_feature_list(self):
        # get the features
        element = list(self.raw_dataset.take(count=1))[0]
        # parse the element in to the example message
        example = tf.train.Example()
        example.ParseFromString(element.numpy())
        print(list(example.features.feature))


    def _parse_function(self, example_proto):
        features = {
            "audio": FixedLenFeature([64000], dtype=tf.float32),
            "instrument_family": FixedLenFeature([1], dtype=tf.int64),
        }
        example = parse_single_example(example_proto, features)

        if self.hparams.create_spectrogram:
            # create spectrogram
            example["spectrogram"] = spectrogram_tensorflow(
                example["audio"], self.hparams
            )
            # create melspectrogram
            example["spectrogram"] = tf.expand_dims(
                tf.transpose(tf.tensordot(
                    example["spectrogram"], self.mel_matrix, 1
                )), axis=2
            )

        cat = example['instrument_family']
        example["category"] = tf.one_hot(tf.squeeze(cat), 11, dtype=tf.int64)
        #print(example["category"])
        return example

    def predicate(self, x, allowed_labels=tf.constant([0,3,4,8,10])):
        label = x["instrument_family"]
        #print(label)
        isAllowed = tf.equal(allowed_labels, tf.cast(label, tf.int32))
        reduced = tf.reduce_sum(tf.cast(isAllowed, tf.int32))
        return tf.greater(reduced, tf.constant(0))

"""### Produce the datasets from tfrecords. """

trainset = NSynthDataset(train_path, hparams)
validset = NSynthDataset(valid_path, hparams)
testset = NSynthDataset(test_path, hparams, is_training=False)


"""### Predict with saved model. """

def predict_with_saved():
    model = load_model(
        'saved_models/2D/2D_keras_nsynth_trained_25_08_2020_14_57_18.h5')
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer =opt, loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    for i, batch in enumerate(validset.dataset):
        result = model.predict(batch['spectrogram'])
        # print(i, batch['spectrogram'])
        print(i, result)
        #yield (batch['spectrogram'], batch['category'])


"""### Test plot an example from the dataset. """

def testplot_example():
    print("Batch size: " + str(hparams.batch_size))
    print('Trainset raw dataset cardinality: '
          + str(tf.data.experimental.cardinality(trainset.raw_dataset)))
    print('Trainset raw dataset size: '
          + str(trainset.raw_dataset.__sizeof__()))
    print('Trainset dataset size: '
          + str(trainset.dataset.__sizeof__()))

    for element in trainset.dataset:
        ex = next(iter(trainset.dataset))
        np.shape(ex["spectrogram"].numpy())

        fig, ax = plt.subplots(ncols=1, figsize=(15, 4))
        cax = ax.matshow(np.squeeze(ex["spectrogram"].numpy()[10]),
                         aspect='auto', origin='lower')
        fig.colorbar(cax)
        plt.show()

        spec_shape = np.shape(ex["spectrogram"].numpy()[10])


"""### Test how fast we can iterate over the dataset. """

def test_iteration_velocity():
    for epoch in range(hparams.num_epochs):
        for batch, train_x in tqdm(
                zip(range(10), trainset.dataset), total=10):
            #print(type(trainset.dataset))
            continue

#test_iteration_velocity()

""""#### Print every batch."""

def print_batch():
    for i, batch in enumerate(validset.dataset):
        #print(i, batch)
        # print(i, 'audio shape' + str(batch['audio'].shape))
        # print(batch['audio'])
        print(i, 'spectrogram shape' + str(batch['spectrogram'].shape))
        #print(batch['spectrogram'])
        # print(i, 'category shape' + str(batch['category'].shape))
        #print(i, batch['category'])
        if i >= 2500:
            break

"""### Create a data generator 
(yielding spectrograms and labels for the 2D network). """

def generate_data(dataset):
    for i, batch in enumerate(dataset):
        yield (batch['spectrogram'], batch['category'])


"""### Define the network. """

#num_epochs, num_classes, batch_size declared in hparams

now = datetime.now()
date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
save_dir = os.path.join(os.getcwd(), 'saved_models/2D/')
model_name = '2D_keras_nsynth_trained_'+ date_time + '.h5'
print("Trained model will be saved as {} in directory {}".format(model_name,
                                                                 save_dir))
print("batch_size="+str(hparams.batch_size))

def get_model_A():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 160, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(hparams.num_classes))
    model.add(Activation('softmax'))

    model.summary()

    return model

def get_model_B():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(128, 160, 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(hparams.num_classes, activation='softmax'))

    model.summary()

    return model

def get_resnet_model():
    model = tf.keras.applications.InceptionResNetV2(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=(128, 160, 3),
        pooling=None, classes=1000, classifier_activation='softmax'
    )

    return model

#model = get_model_A()
model = get_model_B()


"""### Plot model architecture. """
#plot_model(model, to_file='model_plot_2d.png',
#           show_shapes=True, show_layer_names=True)

# initiate ADAM optimizer
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None,
                            decay=0.0, amsgrad=False)
#opt = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999,
#                            epsilon=None, decay=0.0, amsgrad=False)
#opt = keras.optimizers.Adam(lr=0.00001)


""""#### Compile the model. """

# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer='adam')

"""### Train the model. """

def train_model(model):
    n_batches_train = int(trainset.train_size/hparams.batch_size)
    n_batches_valid = int(validset.valid_size/hparams.batch_size)
    print("Train batches number: " + str(n_batches_train))
    print("Info: It will take a while after each epoch "
          "for it to actually complete.")


    history = model.fit(generate_data(trainset.dataset),
              #batch_size=hparams.batch_size,
              epochs=hparams.num_epochs,
              steps_per_epoch= n_batches_train,
              validation_data=generate_data(validset.dataset),
              validation_steps= n_batches_valid,
              shuffle=True     #Has no effect when steps_per_epoch is not None.
              )
    return history



""""#### Train from a checkpoint of a saved model. """

def train_saved_model():
    n_batches_train = int(trainset.train_size/hparams.batch_size)
    n_batches_valid = int(validset.test_size/hparams.batch_size)
    print("Train batches number: " + str(n_batches_train))
    print("Info: It will take a while after each epoch "
          "for it to actually complete.")
    print("batch_size="+str(hparams.batch_size))

    filepath = "saved_models/2D/2D_keras_nsynth_trained_25_08_2020_14_57_18.h5"
    loaded_model = load_model(filepath)

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history = loaded_model.fit(generate_data(trainset.dataset),
              #batch_size=hparams.batch_size,
              epochs=hparams.num_epochs,
              steps_per_epoch= n_batches_train,
              validation_data=generate_data(validset.dataset),
              validation_steps= n_batches_valid,
              shuffle=True,
              callbacks=callbacks_list
              #data_format='channels_first',
              #padding='same'
              )
    return history

history = train_model(model)

# Do not specify the batch_size if your data is in the form of datasets,
# generators, or keras.utils.Sequence instances (since they generate batches).
# If x is a dataset, generator, or keras.utils.Sequence instance,
# y should not be specified (since targets will be obtained from x).

def plot_acc_loss(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('saved_plots/2D/' + date_time + '_accuracy.png')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('saved_plots/2D/' + date_time + '_loss.png')
    plt.show()

plot_acc_loss(history)

""""### Save and score the model. """

def save_model(model):
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

def score_model(model):
    # Score trained model.
    scores = model.evaluate(generate_data(testset.dataset), verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

save_model(model)
score_model(model)

