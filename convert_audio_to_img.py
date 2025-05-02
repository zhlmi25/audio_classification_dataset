import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ========== CONFIGURATION ==========
AUDIO_DIR = 'C:/Users/User/Desktop/assingment/audio_files/all'       # Folder containing input .wav files
IMAGE_DIR = 'C:/Users/User/Desktop/assingment/dataset/dataset/Allahuakbar'  # Output folder for spectrogram images
SAMPLE_RATE = 16000             # Audio sample rate
N_MELS = 128                    # Number of Mel bands
FIG_SIZE = (3, 3)               # Size of the image in inches

# ========== CREATE OUTPUT FOLDER IF NOT EXIST ==========
os.makedirs(IMAGE_DIR, exist_ok=True)

# ========== FUNCTION TO CONVERT AND SAVE ==========

def audio_to_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(S_dB, sr=sr, cmap='gray')  # Changed to 'gray' for greyscale
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ========== LOOP THROUGH ALL AUDIO FILES ==========

for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        input_path = os.path.join(AUDIO_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(IMAGE_DIR, output_filename)
        audio_to_spectrogram(input_path, output_path)
        print(f"Saved: {output_filename}")