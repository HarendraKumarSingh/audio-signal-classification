# Audio Signal Classification (similar to image classifier)

## 1. Audio data manipulation:

Before starting audio signal classification, first understand the audio data type. Start your prctice by first manipulating an audio file with librosa, aubio and python.


Use the <b>audio-manipulation-with-python.ipynb</b> for audio signal manipulation and plot audio pitches.</br> 

Input audio file: <b>sample.wav</b>,</br> 
output audio file: <b>robot_embedded_output.wav</b></br>

Run the jupyter notebook file and pass input and output file name. The code will retun a robot voice embedded output file to the human input voice file. 

## 2. Audio Signal Classification:
Execute the feature-extraction-train-and-inference.py
</br>
Maintain a folder structure like below for Audio Signal Classification:

audio-signal-classification</br>
├── feature-extraction-training-validation-inference.py</br>
├── data</br>
│   ├── inference</br>
│   │   ├── Inference</br>
│   │   │   ├── 8678.wav</br>
│   │   │   ├── 8679.wav</br>
│   │   │   └── 8681.wav</br>
│   │   └── inference.csv</br>
│   ├── test</br>
│   │   ├── Test</br>
│   │   │   ├── 8043.wav</br>
│   │   │   ├── 8046.wav</br>
│   │   │   └── 8047.wav</br>
│   │   └── test.csv</br>
│   └── train</br>
│       ├── Train</br>
│       │   ├── 0.wav</br>
│       │   ├── 1.wav</br>
│       │   └── 2.wav</br>
│       └── train.csv</br>
├── audio_model.h5</br>
└── predict_output.csv</br>

Audio signal classification trained on below 10 audio classes :</br> 
air_conditioner[0], car_horn[1], children_playing[2], dog_bark[3], drilling[4], engine_idling[5], gun_shot[6], jackhammer[7], siren[8] and street_music[9].</br>

Put all your data into data directory. Keep the file_name and class_name in a <b>.csv</b> file (like <b>train.csv</b> or <b>test.csv</b>).</br>

Write a seperate inference file to check on any input sample. All training, test and inference logic provided in a single file '<b>feature-extraction-training-validation-inference.py</b>', make it modular as per your use.</br>

<b>predict_output.csv</b> will be the output after executing <b>feature-extraction-training-validation-inference.py</b>, this will contain class wise classification prediction probability.</br>
