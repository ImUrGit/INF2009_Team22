import os
import sys
import wave
import json
import mysql.connector
import requests
from vosk import Model, KaldiRecognizer
import time
import pyaudio
import torch
import numpy as np
import pandas as pd
import cv2
import threading
import joblib
from deepface import DeepFace
from openvino import Core
from rnnoise_wrapper import RNNoise
from PIL import Image
import time



##########################################
# Shared result and lock for integration #
##########################################
final_result = {
    "age": None,
    "gender": None,
    "keywords": []  # collected from speech analysis
}
result_lock = threading.Lock()
data_condition = threading.Condition()


#######################################
# Speech Analysis (Vosk + Silero VAD) #
#######################################

class SpeechAnalysisModule:
    def __init__(self, asr_model_path = "vosk-model-small-en-us-0.15", vad_model_dir = "silero_models", keyword_file="list_of_words.txt", rnnoise_path="rpi-rnnoise/rnnoise_mono.lv2/librnnoise_mono.so", noise_filter=False):
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
        max_buffer = b'' # If person keeps talking, we limit it
        
        self.check_against_list = self.keywords.copy()
        
        print("Checking against:", self.check_against_list)
        
        # start = time.time()
        while True:
            if not is_listening:
                data = stream.read(512) 
                audio_int16 = np.frombuffer(data, np.int16)
                audio_float32 = self.int2float(audio_int16)
                # vad_start = time.time()
                speech_prob = self.vad_model(torch.from_numpy(audio_float32), 16000).item()
            # print("VAD took:", time.time() - vad_start)
            
            if not is_listening:
                old_data.append(data)
                if len(old_data) > 2:
                    old_data.pop(0)
            
            # print(speech_prob)
            # if True:
            if speech_prob > 0.5 or is_listening:
                data = stream.read(8000) 
                audio_input = data
                if len(old_data) > 0:
                    # Join with frames that occurred before VAD
                    audio_input = audio_input.join(old_data)
                    old_data = []
                if self.noise_filter:
                    audio_input = self.denoiser.filter(audio_input, sample_rate=16000)
                
                # Trim the speech length of talkative people
                if (len(max_buffer) > len(data) * 31.25 * 2):
                    # print("Trimming talkative people")
                    # max_buffer = max_buffer[:len(data) * 16]
                    # print(len(max_buffer), "@", len(data) * 31.25, "per second")
                    # audio_input = max_buffer
                    # recognizer = KaldiRecognizer(self.asr_model,16000) #lightweight intialization
                    max_buffer = b''
                    
                # asr_start = time.time()
                if recognizer.AcceptWaveform(audio_input):
                    # print("ASR took",time.time() - asr_start, "seconds")
                    result = recognizer.Result()
                
                    # print(result)
                    print("complete:", json.loads(result)["text"])
                    is_listening = False
                    max_buffer = b''
                    # recognizer = KaldiRecognizer(self.asr_model,16000)
                    self.send_keyword(results_list)  
                    
                    self.check_against_list = self.keywords.copy()
                    # print("Reset", self.check_against_list)
                    with data_condition:
                        data_condition.notify_all()
                else:
                    # print("ASR took",time.time() - asr_start, "seconds")
                    is_listening = True
                    result = recognizer.PartialResult()
                    results_list = json.loads(result)["partial"]
                    # print("partial:", json.loads(result)["partial"])
                  
                    self.send_keyword(results_list)  
                    with data_condition:
                        data_condition.notify_all()
                
                    
            else:
                # print(speech_prob)
                pass




##############################################
# Override SpeechAnalysisModule.send_keyword #
##############################################
def patched_send_keyword(self, results_list):
    # Call original functionality
    found = []
    for r in self.check_against_list:
        if r in results_list:
            print("Detected (patched):", r)
            found.append(r)
    for keyword in found:
        self.check_against_list.remove(keyword)
    # Update shared result with detected keywords
    with result_lock:
        final_result["keywords"].extend(found)

    with data_condition:
        data_condition.notify_all()


#########################################
# IMAGE ANALYSIS (OpenVINO + DeepFace)
#########################################
def image_analysis():
    core = Core()
    model = core.read_model("/home/grp22/models/res_ssd_300Dim.xml")
    compiled_model = core.compile_model(model, "CPU")
    output_layer = compiled_model.output(0)
    
    cap = cv2.VideoCapture(0)
    face_timers = {}
    PROCESS_DELAY = 1.0  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        input_frame = cv2.resize(frame, (300, 300))
        input_blob = input_frame.transpose(2, 0, 1)[np.newaxis, ...]
        current_time = time.time()
        new_face_timers = {}

        start_inference = time.time()
        detections = compiled_model([input_blob])[output_layer]
        inference_time = time.time() - start_inference

        for detection in detections[0, 0]:
            confidence = detection[2]
            if confidence > 0.7:
                xmin = max(int(detection[3] * w), 0)
                ymin = max(int(detection[4] * h), 0)
                xmax = min(int(detection[5] * w), w - 1)
                ymax = min(int(detection[6] * h), h - 1)
                face_id = (xmin // 10, ymin // 10, xmax // 10, ymax // 10)

                if face_id in face_timers:
                    elapsed_time = current_time - face_timers[face_id]
                    if elapsed_time >= PROCESS_DELAY:
                        face_roi = frame[ymin:ymax, xmin:xmax]
                        try:
                            analysis = DeepFace.analyze(face_roi, actions=['age', 'gender'], enforce_detection=False)
                            if isinstance(analysis, list):
                                analysis = analysis[0]
                            age = analysis['age']
                            gender = analysis['dominant_gender'].capitalize()
                            with result_lock:
                                final_result["age"] = age
                                final_result["gender"] = gender
                            text = f"{gender}, Age: {age}"
                            cv2.putText(frame, text, (xmin, ymin - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            print(f"[Image] Inference: {inference_time:.4f} sec | Result: Age={age}, Gender={gender}")
                        except Exception as e:
                            print("Error analyzing face:", e)
                else:
                    face_timers[face_id] = current_time
                new_face_timers[face_id] = face_timers.get(face_id, current_time)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        face_timers = new_face_timers
        cv2.imshow("Image Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


#########################################
# DECISION MODULE (Combine & Predict)
#########################################
# Load decision module pipeline and label encoder
pipeline = joblib.load('/home/grp22/SpeechAnalysisModule/pickles/model_pipeline.pkl')
label_encoder = joblib.load('/home/grp22/SpeechAnalysisModule/pickles/label_encoder.pkl')


def get_random_product_image(category):
    try:
        # Update these connection parameters with your actual database credentials
        cnx = mysql.connector.connect(
            user='user',
            password='Password123!',
            host='192.168.122.3',
            database='inf2009'
        )
        cursor = cnx.cursor()
        query = """
            SELECT product.id, product.image 
            FROM product 
            WHERE product.category = %s 
            ORDER BY RAND() 
            LIMIT 1
        """
        cursor.execute(query, (category,))
        row = cursor.fetchone()
        if row:
            product_id = row[0]
            blob_data = row[1]
            # Convert the BLOB to a numpy array and decode it to an image
            nparr = np.frombuffer(blob_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print("cv2.imdecode failed. The blob data may not be in a supported image format.")
                return None
            else:
                print("Image decoded successfully. Shape:", img.shape)
                return product_id, img

        else:
            print("No product found for category:", category)
            return None
    except mysql.connector.Error as err:
        print("Database error:", err)
        return None
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'cnx' in locals():
            cnx.close()



def decision_module_loop():
    # Instead of a fixed sleep, wait on the condition with a timeout to produce uniform intervals. 
    INTERVAL = 5
    last_prediction_time = time.time()
    while True:
        # time.sleep(5)
        with data_condition:
            data_condition.wait(timeout=INTERVAL)
        current_time = time.time()
        if current_time - last_prediction_time < INTERVAL:
            continue  # Skip prediction until the full window has passed
        last_prediction_time = current_time
        with result_lock:
            age = final_result["age"]
            gender = final_result["gender"]
            keywords = final_result["keywords"].copy()
            # Optionally, clear keywords after processing
            final_result["keywords"] = []
        if age is not None and gender is not None and keywords:
            gender_short = "F" if gender.lower().startswith("f") else "M"
            sample_df = pd.DataFrame({
                "age": [age],
                "gender": [gender_short],
                "keywords": [", ".join(keywords)]
            })
            # Print out the data fed into the model for verification
            print("[Decision] Data fed to model:")
            print(sample_df)
            try:
                predicted_category_encoded = pipeline.predict(sample_df)
                predicted_category_array = label_encoder.inverse_transform(predicted_category_encoded)
                predicted_category = predicted_category_array[0]
                print("[Decision] Final Predicted Category:", predicted_category)


                # --- Database lookup and image preview
                # Retrieve and preview the product image
                product_id, product_img = get_random_product_image(predicted_category)
                if product_img is not None:
                    print(f"Found product ID: {product_id} for category: {predicted_category}")
                    print("Image decoded successfully. Shape:", product_img.shape)


                    # Call the Flask route to update the DB for this product
                    update_url = f"http://192.168.122.230:5000/update/{product_id}"
                    try:
                        response = requests.put(update_url)
                        if response.status_code == 200:
                            print("Successfully updated DB for product ID:", product_id)
                        else:
                            print("Failed to update DB. Status code:", response.status_code)
                    except Exception as e:
                        print("Error calling Flask route:", e)

                    # Convert the OpenCV image (BGR) to a PIL image (RGB)
                    pil_img = Image.fromarray(cv2.cvtColor(product_img, cv2.COLOR_BGR2RGB))
                    pil_img.show()
                    time.sleep(5)
                    # Note: .show() is non-blocking; 
                else:
                    print("No product image available for predicted category.")
                # -------------------------------------------

            except Exception as e:
                print("Error in decision module:", e)

    

########################################
# MAIN: Start all threads concurrently #
########################################
if __name__ == "__main__":
    # Instantiate the speech module
    sam = SpeechAnalysisModule(noise_filter=False)
    # Monkey-patch its send_keyword method without changing the original file
    sam.send_keyword = patched_send_keyword.__get__(sam, SpeechAnalysisModule)

    # Start threads for image analysis, speech analysis, and decision module
    image_thread = threading.Thread(target=image_analysis, daemon=True)
    speech_thread = threading.Thread(target=sam.listen, daemon=True)
    decision_thread = threading.Thread(target=decision_module_loop, daemon=True)

    image_thread.start()
    speech_thread.start()
    decision_thread.start()


    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("Exiting...")
