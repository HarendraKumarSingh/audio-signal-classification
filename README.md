# Audio Signal Classification (similar to image classifier)

## Audio data manipulation:

#1. Before starting audio signal classification, first understand the audio data type. Start your prctice by first manipulating an audio file with librosa, aubio and python.


Use the <b>audio-manipulation-with-python.ipynb</b> for audio signal manipulation and plot audio pitches.
Input audio file: <b>sample.wav</b>
output audio file: <b>robot_embedded_output.wav</b>
Run the jupyter notebook file and pass input and output file name. The code will retun a robot voice embedded output file to the human input voice file. 

### Audio Signal Classification:
#2.Execute the feature-extraction-train-and-inference.py
</br>
Maintain a folder structure like below for Audio Signal Classification:

audio-signal-classification
├── audio_feature_extractor.py
├── audio_model_0_5441.h5
├── audio_model.h5
├── data
│   ├── inference
│   │   ├── Inference
│   │   │   ├── 8678.wav
│   │   │   ├── 8679.wav
│   │   │   └── 8681.wav
│   │   └── inference.csv
│   ├── test
│   │   ├── Test
│   │   │   ├── 8043.wav
│   │   │   ├── 8046.wav
│   │   │   └── 8047.wav
│   │   └── test.csv
│   └── train
│       ├── Train
│       │   ├── 0.wav
│       │   ├── 1.wav
│       │   └── 2.wav
│       └── train.csv
├── feature-extraction-train-and-inference.py
└── predict_output.csv

Put all your data into data directory. Keep the file_name and class_name in a <b>.csv</b> file (like <b>train.csv</b> or <b>test.csv</b>).
Write a seperate inference file to check on any input sample. All training, test and inference logic provided in a single file '<b>feature-extraction-train-and-inference.py</b>', make it modular as per your use.
<b>predict_output.csv</b> will be the output after executing <b>feature-extraction-train-and-inference.py</b>, this will contain class wise classification prediction probability.
