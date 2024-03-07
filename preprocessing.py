import os
import numpy as np
from glob import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import gc

train_data_path = 'data\\train\\'
test_data_path = 'data\\test\\'
wav_path = 'data\\wav\\'

def create_spectrogram(filename, name, file_path):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    
    fig, ax = plt.subplots(figsize=[0.72, 0.72])
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    
    filename = os.path.join(file_path, name + '.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    
    # Close the figure properly
    plt.close(fig)
    
    del filename, name, clip, sample_rate, fig, ax, S

# Lap trong thu muc data/wav/train va tao ra 4000 file anh spectrogram
Data_dir = np.array(glob(os.path.join(wav_path, "train\\*")))

for file in Data_dir[0:4000]:
    filename, name = file, file.split('\\')[-1].split('.')[0]
    create_spectrogram(filename, name, train_data_path)

gc.collect()

# Lap trong thu muc data/wav/test va tao ra 3000 file anh spectrogram
Test_dir = np.array(glob(os.path.join(wav_path, "test\\*")))

for file in Test_dir[0:3000]:
    filename, name = file, file.split('\\')[-1].split('.')[0]
    create_spectrogram(filename, name, test_data_path)

gc.collect()

print("Process done!")
