from config import *
import os
import sys

from tensorflow_datasets.core.utils import tqdm
import tensorflow as tf
from tensorflow.io import FixedLenFeature, parse_single_example

import keras
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Input
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

from librosa.core.time_frequency import mel_frequencies
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

print(tf.__version__)
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
sess = tf.compat.v1.Session(config=config)
#tf.compat.v1.disable_eager_execution()


#current datasets paths
train_path = [str(train_tf_prep_path)]#, str(train_tf_prep_aug_path)]
valid_path = [str(valid_tf_prep_path)]#, str(valid_tf_prep_aug_path)]
test_path = [str(test_tf_prep_path)]#, str(test_tf_prep_aug_path)]

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
    num_epochs = 43,
    num_classes = 11,
    # spectrogramming
    sample_rate = 16000,
    create_spectrogram = False,
    win_length = 1024,
    n_fft = 1024,
    hop_length= 400,
    ref_level_db = 50,
    min_level_db = -100,
    # mel scaling
    num_mel_bins = 128,
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
            binary_classification=False,
            prefetch=1000,  # how many spectrograms to prefetch
            num_parallel_calls=10,  # how many threads should be preparing data
            n_samples=305979,  # how many items are in the dataset
            train_size=289205,
            valid_size=12678,
            test_size=4096,
            shuffle_buffer=hparams.batch_size * 50,
    ):
        self.vocal_label = self.const_tensor([10])
        self.binary_classification = binary_classification
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
        ### filter out unwanted labels
        #self.dataset = self.dataset.filter(self.predicate)
        ### make and split train and test datasets
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
        # Slaney-style mel is scaled to be approx constant energy per channel
        # (from librosa)
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

    def const_tensor(self, x):
        return tf.constant(x, dtype=tf.int64)

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

        if self.binary_classification:
            cat = example['instrument_family']
            example["category"] = tf.cond(tf.math.equal(cat, self.vocal_label),
                                          lambda: tf.constant([1]),
                                          lambda: tf.constant([0]))
        else:
            cat = example['instrument_family']
            example["category"] = tf.one_hot(tf.squeeze(cat), 11)
        return example

    def predicate(self, x, allowed_labels=tf.constant([0,3,4,8,10])):
        label = x["instrument_family"]
        #print(label)
        isAllowed = tf.equal(allowed_labels, tf.cast(label, tf.int32))
        reduced = tf.reduce_sum(tf.cast(isAllowed, tf.int32))
        print(label)
        return tf.greater(reduced, tf.constant(0))

"""### Produce the datasets from tfrecords. """

trainset = NSynthDataset(train_path, hparams)
validset = NSynthDataset(valid_path, hparams)
testset = NSynthDataset(test_path, hparams, is_training=False)


"""### Predict with saved model. """

def predict_with_saved():
    model = load_model(
            'saved_models/1D/1D_keras_nsynth_trained_10_09_2020_10_52_35.h5')
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer =opt, loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    for i, batch in enumerate(validset.dataset):
        result = model.predict(batch['audio'])
        # print(i, batch['spectrogram'])
        print(i, result)
        #yield (batch['spectrogram'], batch['category'])

""" ### Predict with binary model. """

# Create a description of the features.
feature_description = {
    "audio": FixedLenFeature([64000], dtype=tf.float32),
    "category": FixedLenFeature([1], dtype=tf.int64)
}

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

def predict_bin_with_saved_A():
    model = load_model(
            'saved_models/1D/1D_keras_nsynth_trained_10_09_2020_10_52_35.h5')
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer =opt, loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    for i, batch in enumerate(testset.dataset):
        #print(batch['audio'])
        result = model.predict(batch['audio'])
        print(i, batch['category'], result)

def predict_bin_with_saved_B():
    model = load_model(
            'saved_models/1D/1D_keras_nsynth_trained_10_09_2020_10_52_35.h5')
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer =opt, loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    raw_dataset = tf.data.TFRecordDataset('recording_2.tfrecord')

    rec_dataset = raw_dataset.map(_parse_function)
    rec_dataset = rec_dataset.batch(93)

    #result = model.predict(rec_dataset['audio'], 21)

    for i, batch in enumerate(rec_dataset):
        #print(batch['audio'])
        result = model.predict(batch['audio'])
        print(i, batch['category'], result)

#predict_bin_with_saved_A()


"""### Calculate ROC AUC score for binary classificator. """

def calculate_roc_auc():
    model = load_model(
        'saved_models/1D/1D_keras_nsynth_trained_10_09_2020_10_52_35.h5')
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer =opt, loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    y_true = np.empty((16,256,1), dtype=int)
    y_pred = np.empty((16,256,1), dtype=float)

    for i, batch in enumerate(testset.dataset):
        proto_tensor = tf.make_tensor_proto(batch['category'])
        y_true[i] = tf.make_ndarray(proto_tensor)
        y_pred[i] = model.predict(batch['audio'])
        #print(i, y_true, y_pred)
        #print(i, batch['spectrogram'])
        #print(i, result)
        #yield (batch['spectrogram'], batch['category'])

    y_true = np.squeeze(y_true).flatten()
    y_pred = np.squeeze(y_pred).flatten()
    np.set_printoptions(threshold=sys.maxsize)
    # print(y_true)
    # print(y_pred)
    #print(y_true, y_pred)
    print(y_true.shape)
    print(y_pred.shape)

    roc_score = roc_auc_score(y_true, y_pred)
    print(roc_score)

#calculate_roc_auc()


"""### Test plot an example from the dataset. """

def testplot_example():
    print("Batch size: " + str(hparams.batch_size))
    print('Trainset raw dataset cardinality: '
          + str(tf.data.experimental.cardinality(trainset.raw_dataset)))
    print('Trainset raw dataset size: ' +
          str(trainset.raw_dataset.__sizeof__()))
    print('Trainset dataset size: ' +
          str(trainset.dataset.__sizeof__()))

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


"""### Check if batches differ from each other. """

def compare_batches():
    prev_batch = None
    for i, batch in enumerate(validset.dataset):
        prev_batch = batch
        if i >= 1:
            break
    for i, batch in enumerate(validset.dataset):
        #print(i, batch)
        # print(i, 'audio shape' + str(batch['audio'].shape))
        # print(batch['audio'])
        print("batch:")
        print(batch["audio"])
        print("prev_batch:")
        print(prev_batch["audio"])
        #print(tf.sets.difference(batch["audio"], prev_batch["audio"]))
        print(tf.equal(batch["audio"], prev_batch["audio"]))
        if i >= 2500:
            break


""""#### Print every batch."""

def print_batch_A():
    for i, batch in enumerate(validset.dataset):
        #print(i, batch)
        # print(i, 'audio shape' + str(batch['audio'].shape))
        # print(batch['audio'])
        # print(i, 'spectrogram shape' + str(batch['spectrogram'].shape))
        # print(batch['spectrogram'])
        print(i, 'category shape' + str(batch['category'].shape))
        print(i, batch['category'])
        if i >= 2500:
            break

def print_batch_B():
    for i, batch in enumerate(validset.dataset):
        for sample in batch['audio']:
            print(sample.shape)
            print(sample)

def print_batch_C():
    for i, batch in enumerate(trainset.dataset):
        for aud, cat in zip(batch['audio'], batch['category']):
            aud_reshaped =  tf.expand_dims(aud, 1)
            cat_reshaped = tf.expand_dims(cat, 1)
            print(aud_reshaped.shape)
            print(aud_reshaped)
            print(cat_reshaped.shape)
            print(cat_reshaped)

"""### Create a data generator 
(yielding waveforms and labels for the 1D network). """

def generate_data(dataset):
    for i, batch in enumerate(dataset):
        yield (batch['audio'], batch['category'])



"""### Define the network. """

#num_epochs, num_classes, batch_size declared in hparams

now = datetime.now()
date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
save_dir = os.path.join(os.getcwd(), 'saved_models/1D/')
model_name = '1D_keras_nsynth_trained_'+ date_time + '.h5'
print("Trained model will be saved as {} in directory {}".format(model_name,
                                                                 save_dir))

def get_model():
    model = Sequential()

    #input_shape is supposed to be (batch_size, steps, input_dim)
    #model.add(Input(shape=(64000, 1)))
    model.add(tf.keras.layers.Reshape((64000, 1),
                                input_shape=(int(hparams.batch_size),64000)))
    print(model.output_shape)
    model.add(Conv1D(16, kernel_size=9, activation='relu', padding="valid", name="1"))
    model.add(Conv1D(16, kernel_size=9, activation='relu', padding="valid", name="2"))
    model.add(MaxPooling1D(pool_size=16))
    model.add(Dropout(0.1))

    model.add(Conv1D(32, kernel_size=3, activation='relu', padding="valid", name="3"))
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding="valid", name="4"))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.1))

    model.add(Conv1D(32, kernel_size=3, activation='relu', padding="valid", name="5"))
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding="valid", name="6"))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.1))


    model.add(Conv1D(256, kernel_size=3, activation='relu', padding="valid", name="7"))
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding="valid", name="8"))
    model.add(MaxPooling1D(pool_size=4))
    #model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', name="9", kernel_regularizer='l2'))
    #kernel_regularizer='l2' added on 28.08
    model.add(Dense(1028, activation='relu', name="10", kernel_regularizer='l2'))
    #kernel_regularizer='l2' added on 2.09
    model.add(Dense(hparams.num_classes, activation='softmax'))

    model.summary()
    return model

def get_model_binary():
    model = Sequential()
    model.add(tf.keras.layers.Reshape((64000, 1),
                                input_shape=(int(hparams.batch_size), 64000)))
    model.add(Conv1D(16, kernel_size=9, activation='relu', padding="valid", name="1"))
    model.add(Conv1D(16, kernel_size=9, activation='relu', padding="valid", name="2"))
    model.add(MaxPooling1D(pool_size=16))
    model.add(Dropout(0.1))

    model.add(Conv1D(32, kernel_size=3, activation='relu', padding="valid", name="3"))
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding="valid", name="4"))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.1))

    model.add(Conv1D(32, kernel_size=3, activation='relu', padding="valid", name="5"))
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding="valid", name="6"))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.1))

    model.add(Conv1D(256, kernel_size=3, activation='relu', padding="valid", name="7"))
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding="valid", name="8"))
    model.add(MaxPooling1D(pool_size=4))
    # model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', name="9", kernel_regularizer='l2'))
    model.add(Dense(1028, activation='relu', name="10", kernel_regularizer='l2'))
    model.add(Dense(1, activation=tf.nn.sigmoid))

    model.summary()
    return model


"""### Plot model architecture. """
#plot_model(model, to_file='model_plot.png',
#           show_shapes=True, show_layer_names=True)


""""### Compile the model. """

# initiate ADAM optimizer
opt = keras.optimizers.Adam(lr=0.0001)

# model =get_model()
# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

model = get_model()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

""""#### Train the model. """

def train_model(model):
    n_batches_train = int(trainset.train_size/hparams.batch_size)
    n_batches_valid = int(validset.test_size/hparams.batch_size)
    print("Train batches number: " + str(n_batches_train))
    print("Info: It will take a while after each epoch "
          "for it to actually complete.")
    print("batch_size="+str(hparams.batch_size))

    history = model.fit(generate_data(trainset.dataset),
              #batch_size=hparams.batch_size,
              epochs=hparams.num_epochs,
              steps_per_epoch= n_batches_train,
              validation_data=generate_data(validset.dataset),
              validation_steps= n_batches_valid,
              shuffle=True,
              #data_format='channels_first',
              #padding='same'
              )

    return history

history = train_model(model)

# Do not specify the batch_size if your data is in the form of datasets,
# generators, or keras.utils.Sequence instances (since they generate batches).
# If x is a dataset, generator, or keras.utils.Sequence instance,
# y should not be specified (since targets will be obtained from x).


""""#### Train from a checkpoint of a saved model. """

def train_saved_model():
    n_batches_train = int(trainset.train_size/hparams.batch_size)
    n_batches_valid = int(validset.test_size/hparams.batch_size)
    print("Train batches number: " + str(n_batches_train))
    print("Info: It will take a while after each epoch "
          "for it to actually complete.")
    print("batch_size="+str(hparams.batch_size))

    filepath = "saved_models/1D/1D_keras_nsynth_trained_10_09_2020_10_52_35.h5"
    new_model = load_model(filepath)

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history = new_model.fit(generate_data(trainset.dataset),
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

#history = train_saved_model()

""""### Plot the data. """

def plot_acc_loss(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('saved_plots/1D/' + date_time + '_accuracy.png')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('saved_plots/1D/' + date_time + '_loss.png')
    plt.show()

plot_acc_loss(history)


""""### Save and score the model. """

def save_model(model):
    #Save model and weights
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