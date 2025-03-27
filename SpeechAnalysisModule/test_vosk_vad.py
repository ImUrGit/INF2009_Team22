import os
import sys
import wave
import json
from vosk import Model, KaldiRecognizer
import time
import pyaudio
import torch
import numpy as np

# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

# Path to the model (adjust according to where you extracted the model)
model_path = "vosk-model-small-en-us-0.15"

vad_model, utils = torch.hub.load(repo_or_dir='silero_models',
                              model='silero_vad',
                              source="local",
                              force_reload=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

#model_path = "vosk-model-en-us-0.22"
# Load the Vosk model
if not os.path.exists(model_path):
    print("Model not found, please make sure to download the model.")
    sys.exit()

model = Model(model_path)

with open("list_of_words.txt", "r") as f:
    keywords = f.read().splitlines()
    
print("Keywords:",keywords)

if __name__ == "__main__":

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
    
    recognizer = KaldiRecognizer(model,16000)
    
    is_listening = False
    old_data = []
    
    check_against_list = keywords
    while True:
        data = stream.read(512)
        #data = stream.read(4000)
        
        audio_int16 = np.frombuffer(data, np.int16)
        audio_float32 = int2float(audio_int16)
        speech_prob = vad_model(torch.from_numpy(audio_float32), 16000).item()
        
        if not is_listening:
            old_data.append(data)
            if len(old_data) > 2:
                old_data.pop(0)
        
        # print(speech_prob)
        # if True:
        if speech_prob > 0.5 or is_listening:
            audio_input = data
            if len(old_data) > 0:
                audio_input = b''.join(old_data)
                old_data = []
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
            
                # print(result)
                print("complete:", json.loads(result)["text"])
                is_listening = False
                check_against_list = keywords
            else:
                is_listening = True
                result = recognizer.PartialResult()
                results_list = json.loads(result)["partial"]
                # print("partial:", json.loads(result)["partial"])
                
                found = []
                for r in check_against_list:
                    if r in results_list:
                        print("Detected:", r) # Replace with queue pushing code
                        found.append(r)
                
                for keyword in found:
                    check_against_list.remove(keyword)
                
        else:
            # print(speech_prob)
            pass

    