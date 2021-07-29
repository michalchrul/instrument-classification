import librosa
import json
import ntpath
import numpy as np
import soundfile as sf
import random
from tqdm import tqdm
from math import sqrt
from config import *

#stuff to extract filename from full path
ntpath.basename("a/b/c")

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

#get 2 first letters of a string
def first2(s):
    return s[:2]

def check_if_vox(filename):
    cat = first2(path_leaf(filename))
    if cat == 'vo':
        return True
    else:
        return False

def loadfile(filename, frames):
    sr = 16000
    y, sr = sf.read(filename, frames=frames)
    #print(y, sr)
    return y

#convolve mono signal with mono ir
def convolve_mono(signal_filename, ir_filename, output_path):
    sr = 16000
    output_filename = output_path / path_leaf(signal_filename)
    print(output_filename)

    sig = loadfile(signal_filename, frames=64000)
    win = loadfile(ir_filename, frames=2048)
    output = np.convolve(sig, win, mode='same')

    sf.write(output_filename, output, sr)
    return output

#convolve mono signal with multichannel ir
def convolve_multi(signal_filename, ir_np_array, output_path):
    sr = 16000
    #adding ir's idx to the filename
    output_filename = output_path / path_leaf(signal_filename)
    output_filename = str(os.path.splitext(output_filename)[0]) \
                      + '_' + str(ir_np_array[0]) + '.wav'

    sig = loadfile(signal_filename, frames=64000)
    output = np.empty([64000, 19])

    for i, channel in enumerate(ir_np_array[1].T):
        win = channel
        mono_conv = np.convolve(sig, win, mode='same')
        output[:, i] = np.squeeze(mono_conv)

    output = np.asfortranarray(output)
    sf.write(output_filename, output, sr)
    return output

#importing IR JSON
def import_ir_json():
    with open(impresp_path / 'constellation.json' ) as constellation_data:
        data = constellation_data.read()
        positions = json.loads(data)
        print(positions)
        for each in positions['essCoordinates']:
            print(each['phi'], each['theta'])


#returns a dictionary of impulse responses [phi][idx][ch]
def load_ir():
    idx_dict = {}
    idx_dict[0] = [0, 418, 421, 420, 423]
    idx_dict[45] = [14, 26, 29, 98, 101]
    idx_dict[135] = [15, 27, 31, 99, 103]
    idx_dict[225] = [17, 30, 33, 102, 105]
    idx_dict[315] = [16, 28, 32, 100, 104]

    imp_res = {}
    for k in idx_dict.keys():
        imp_res[k] = {}
        for idx in idx_dict[k]:
            y = loadfile(str(impresp_path) + "/" + str(idx) + ".wav",
                         frames=2048)
            imp_res[k][idx] = y
        # print(k)
        #print(imp_res[k])

    # accessing the 0 channel of a impulse response with idx 16 for phi = 315
    # print(imp_res[315][16][0])
    return imp_res

#convolve all of the samples from a dataset
def convolve_dataset(dataset_path, output_path):
    for filename in tqdm(os.listdir(dataset_path)):
        file_path = dataset_path / filename
        if check_if_vox(filename):
            random_ir = random.choice(list(imp_res[0].items()))
            convolve_multi(file_path, random_ir, output_path)
        else:
            phi = random.choice([45, 135, 225, 315])
            random_ir = random.choice(list(imp_res[phi].items()))
            convolve_multi(file_path, random_ir, output_path)
    print("Successfully convolved samples from " \
          + str(dataset_path) + " and saved them to " \
          + str(output_path))

def mix_samples_test(sample_A_path,  sample_B_path, sample_C_path):
    sample_A = loadfile(sample_A_path, 64000)
    sample_B = loadfile(sample_B_path, 64000)
    sample_C = loadfile(sample_C_path, 64000)
    sum = sample_A + sample_B + sample_C
    output_filename = convoluted_path / 'mix_check' / 'mixdown.wav'
    output = np.empty([192000, 19])

    for i, channel in enumerate(sum.T):
        output[:, i] = librosa.core.resample(
            np.asfortranarray(channel), 16000, 48000,
            res_type='kaiser_best', fix=True)

    output = np.asfortranarray(output)
    sf.write(output_filename, output, 48000)

#mix samples to simulate their co-presence in a virtual scene
def mixdown_dataset(dataset_path, output_path):
    list_of_files = os.listdir(dataset_path)
    for filename in tqdm(list_of_files):
        file_path = dataset_path / filename
        #sample_A = loadfile(file_path, 64000)

        sample_A, sr = librosa.load(file_path, mono=True, sr=16000, duration=4)
        sample_A = np.expand_dims(sample_A, 1)

        rms_A = librosa.feature.rms(sample_A, frame_length=64000)

        max = random.randint(2,5)
        for i in range(1, max):
            random_file = random.choice(
                            [ele for ele in list_of_files if ele != filename])
            random_path = dataset_path / random_file
            #sample_B = loadfile(random_path, 64000)

            sample_B, sr = librosa.load(random_path, mono=True, sr=16000,
                                        duration=4)
            sample_B = np.expand_dims(sample_B, 1)

            rms_B = librosa.feature.rms(sample_B, frame_length=64000)
            ratio = 0.10 / sqrt(rms_B/rms_A)
            sample_A += ratio * sample_B

        output_filename = output_path / filename
        sf.write(output_filename, sample_A, 16000)

    print("Successfully mixed samples from " \
          + str(dataset_path) + " and saved them to " \
          + str(output_path))

#mix samples to simulate their co-presence in a virtual scene
#with data augmentation
def mixdown_dataset_aug(dataset_path, output_path):
    list_of_files = os.listdir(dataset_path)
    iterations = 4
    for iter in range(iterations):
        for filename in tqdm(list_of_files):
            if (check_if_vox(filename)):
                file_path = dataset_path / filename
                #sample_A = loadfile(file_path, 64000)

                sample_A, sr = librosa.load(file_path, mono=True, sr=16000,
                                            duration=4)
                sample_A = np.expand_dims(sample_A, 1)

                rms_A = librosa.feature.rms(sample_A, frame_length=64000)

                max = random.randint(2,5)
                for i in range(1, max):
                    random_file = random.choice(
                                    [ele for ele in list_of_files
                                     if ele != filename])
                    random_path = dataset_path / random_file
                    #sample_B = loadfile(random_path, 64000)

                    sample_B, sr = librosa.load(random_path, mono=True,
                                                sr=16000, duration=4)
                    sample_B = np.expand_dims(sample_B, 1)

                    rms_B = librosa.feature.rms(sample_B, frame_length=64000)
                    ratio = 0.10 / sqrt(rms_B/rms_A)
                    sample_A += ratio * sample_B

                filename = str(os.path.splitext(filename)[0]) \
                      + '_' + str(iter) + '.wav'
                output_filename = output_path / filename
                sf.write(output_filename, sample_A, 16000)

    print("Successfully mixed samples from " \
          + str(dataset_path) + " and saved them to " \
          + str(output_path))


imp_res = load_ir()

# convolve_dataset(trainset_path, conv_train_path)
# convolve_dataset(validset_path, conv_valid_path)
# convolve_dataset(testset_path, conv_test_path)
#
# mixdown_dataset(conv_train_path, mix_train_path)
# mixdown_dataset(conv_valid_path, mix_valid_path)
# mixdown_dataset(conv_test_path, mix_test_path)
#
# mixdown_dataset_aug(conv_valid_path, mix_train_aug_path)
# mixdown_dataset_aug(conv_train_path, mix_valid_aug_path)
# mixdown_dataset_aug(conv_train_path, mix_test_aug_path)