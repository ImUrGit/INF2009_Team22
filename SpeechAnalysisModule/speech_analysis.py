import os
import sys
import wave
import json
from vosk import Model, KaldiRecognizer
import time
import pyaudio
import torch
import numpy as np
from rnnoise_wrapper import RNNoise
import time

class SpeechAnalysisModule:
    def __init__(self, asr_model_path = "vosk-model-small-en-us-0.15", vad_model_dir = "silero_models", keyword_file="list_of_words.txt", rnnoise_path="linux-rnnoise/rnnoise_mono.lv2/librnnoise_mono.so", noise_filter=False):
        # Path to the model (adjust according to where you extracted the model)
        
        self.vad_model, self.utils = torch.hub.load(repo_or_dir=vad_model_dir,
                              model='silero_vad',
                              source="local",
                              force_reload=True)

        self.p = pyaudio.PyAudio()
        
        self.noise_filter = noise_filter
        if self.noise_filter:
            self.denoiser = RNNoise(rnnoise_path)
        
        (self.get_speech_timestamps,
        self.save_audio,
        self.read_audio,
        self.VADIterator,
        self.collect_chunks) = self.utils
        
        # Load the Vosk model
        self.asr_model = Model(asr_model_path)
        
        if not os.path.exists(asr_model_path):
            print("ASR Model not found")
            sys.exit()
        
        with open(keyword_file, "r") as f:
            self.keywords = f.read().splitlines()
        
        
    # Provided by Alexander Veysov
    def int2float(self,sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/32768
        sound = sound.squeeze()  # depends on the use case
        return sound
    
    def send_keyword(self, results_list):
        found = []
        for r in self.check_against_list:
            if r in results_list:
                print("Detected:", r) # Replace with queue pushing code
                found.append(r)
        
        for keyword in found:
            self.check_against_list.remove(keyword)
    
    def listen(self):
        stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        recognizer = KaldiRecognizer(self.asr_model,16000)
        
        is_listening = False
        old_data = []
        
        self.check_against_list = self.keywords
        
        print("Checking against:", self.check_against_list)
        
        while True:
            data = stream.read(512)
            #data = stream.read(4000)
            
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = self.int2float(audio_int16)
            # vad_start = time.time()
            speech_prob = self.vad_model(torch.from_numpy(audio_float32), 16000).item()
            # print("VAD took," time.time() - vad_start,"seconds")
            
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
                if self.noise_filter:
                    audio_input = self.denoiser.filter(audio_input, sample_rate=16000)
                # asr_start = time.time()
                if recognizer.AcceptWaveform(audio_input):
                    # print("ASR took," time.time() - asr_start,"seconds")
                    result = recognizer.Result()
                
                    # print(result)
                    print("complete:", json.loads(result)["text"])
                    is_listening = False
                    
                    self.send_keyword(results_list)  
                    
                    self.check_against_list = self.keywords
                else:
                    # print("ASR took," time.time() - asr_start,"seconds")
                    is_listening = True
                    result = recognizer.PartialResult()
                    results_list = json.loads(result)["partial"]
                    # print("partial:", json.loads(result)["partial"])
                  
                    self.send_keyword(results_list)  
                
                    
            else:
                # print(speech_prob)
                pass


if __name__ == "__main__":

    # sam = SpeechAnalysisModule(rnnoise_path="macos-rnnoise/rnnoise_stereo.lv2/librnnoise_stereo.so",noise_filter=True)
    sam = SpeechAnalysisModule(noise_filter=True)
    sam.listen()

    