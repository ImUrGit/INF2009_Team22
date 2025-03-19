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

# Function to recognize speech from a file
def recognize_audio(audio_file):
    wf = wave.open(audio_file, "rb")

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        print("Audio file must be mono PCM WAV at 16kHz.")
        sys.exit()

    recognizer = KaldiRecognizer(model, wf.getframerate())

    results = []
    p_text_lst = []
    start = time.time()
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            print("End")
            break
        print("Frame of 4000")
        if recognizer.AcceptWaveform(data):
            print("Accepted")
            result = recognizer.Result()
            results.append(json.loads(result))
        else:
            print("Partial")
            result = recognizer.PartialResult()
            p_text_lst.append(result)

    print("Elapsed time:",time.time()-start)
    print("Results:")
    # print(results)
    # print(p_text_lst)
    # Output recognized text
    for result in results:
        print(result.get('text', ''))
    for result in p_text_lst:
        print(result["partial"])



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
            # print(result)
            # print(result)
            print("partial:", json.loads(result)["partial"])
        

#old code for speech recognition from file
if __name__ == "__main__":
    # Provide the path to your WAV audio file
    
    audio_file = "audio_curry_powder.wav"  # Change to your file path
    
    recognize_audio(audio_file)
    