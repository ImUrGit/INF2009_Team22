import os
import sys
import wave
import json
from vosk import Model, KaldiRecognizer
import time
import pyaudio

# Path to the model (adjust according to where you extracted the model)
model_path = "vosk-model-small-en-us-0.15"

#model_path = "vosk-model-en-us-0.22"
# Load the Vosk model
if not os.path.exists(model_path):
    print("Model not found, please make sure to download the model.")
    sys.exit()

model = Model(model_path)

with open("list_of_words.txt", "r") as f:
    keywords = f.readlines()

if __name__ == "__main__":

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
    
    recognizer = KaldiRecognizer(model,16000)
    
    while True:
        data = stream.read(4000)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            # print(result)
            print("complete:", json.loads(result)["text"])
        else:
            result = recognizer.PartialResult()
            print("partial:", json.loads(result)["partial"])

    