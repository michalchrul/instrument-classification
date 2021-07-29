import os
from pathlib import Path
from pathlib import PosixPath

PROJECT_PATH = Path(os.path.realpath('config.py')).parent.parent

# main directories:
datasets_path = PROJECT_PATH / 'datasets'
convoluted_path = PROJECT_PATH / 'convoluted'
hdd_path = PosixPath('/media/michal/HDD/')
impresp_path = datasets_path / 'impresp'
ssd_path = PosixPath('/media/michal/ef510800-87c3-4021-ada5-2db57ae240b1/home/michal/Datasets')

### Paths for model input:

# original audio files:
trainset_audio_path = datasets_path / 'nsynth-train' / 'audio'
validset_audio_path = datasets_path / 'nsynth-valid' / 'audio'
testset_audio_path = datasets_path / 'nsynth-test' / 'audio'

# original tfrecord files:
trainset_tf_path = datasets_path / 'nsynth-train'/ 'nsynth-train.tfrecord'
validset_tf_path = datasets_path / 'nsynth-valid' / 'nsynth-valid.tfrecord'
testset_tf_path = datasets_path / 'nsynth-test' / 'nsynth-test.tfrecord'

#hdd path
# # preprocessed tfrecord files:
# train_tf_prep_path = hdd_path / 'nsynth-train-prep.tfrecord'
# valid_tf_prep_path = hdd_path / 'nsynth-valid-prep.tfrecord'
# test_tf_prep_path = hdd_path / 'nsynth-test-prep.tfrecord'
#
# # preprocessed tfrecord files (augmented vocals only):
# train_tf_prep_aug_path = hdd_path / 'nsynth-train-prep-aug.tfrecord'
# valid_tf_prep_aug_path = hdd_path / 'nsynth-valid-prep-aug.tfrecord'
# test_tf_prep_aug_path = hdd_path / 'nsynth-test-prep-aug.tfrecord'

#ssd path
# preprocessed tfrecord files:
train_tf_prep_path = ssd_path / 'nsynth-train-prep.tfrecord'
valid_tf_prep_path = ssd_path / 'nsynth-valid-prep.tfrecord'
test_tf_prep_path = ssd_path / 'nsynth-test-prep.tfrecord'

# preprocessed tfrecord files (augmented vocals only):
train_tf_prep_aug_path = ssd_path / 'nsynth-train-prep-aug.tfrecord'
valid_tf_prep_aug_path = ssd_path / 'nsynth-valid-prep-aug.tfrecord'
test_tf_prep_aug_path = ssd_path / 'nsynth-test-prep-aug.tfrecord'


### Directories for preprocessing: ###

# convoluted datsets paths:
conv_train_path = PosixPath('/media/michal/HDD/nsynth-train-conv')
conv_valid_path = convoluted_path / 'nsynth-valid-conv'
conv_test_path = convoluted_path / 'nsynth-test-conv'
conv_other_path = convoluted_path / 'other-conv'

# mixed datasets paths:
mix_train_path = PosixPath('/media/michal/HDD/nsynth-train-mix')
mix_valid_path = PosixPath('/media/michal/HDD/nsynth-valid-mix')
mix_test_path = PosixPath('/media/michal/HDD/nsynth-test-mix')
mix_other_path = convoluted_path / 'other-mix'

# mixed data set paths (vocal augmentations only):
mix_train_aug_path = PosixPath('/media/michal/HDD/nsynth-train-mix-aug')
mix_valid_aug_path = PosixPath('/media/michal/HDD/nsynth-valid-mix-aug')
mix_test_aug_path = PosixPath('/media/michal/HDD/nsynth-test-mix-aug')
