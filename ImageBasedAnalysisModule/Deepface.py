import cv2
import time
import numpy as np
from deepface import DeepFace
from openvino.runtime import Core

# Initialize OpenVINO Runtime
core = Core()
model = core.read_model("/home/icepi/models/res_ssd_300Dim.xml")
compiled_model = core.compile_model(model, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Video capture from Pi camera or USB camera
cap = cv2.VideoCapture(0)

face_timers = {}
PROCESS_DELAY = 2.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Prepare input for model
    input_frame = cv2.resize(frame, (300, 300))
    input_blob = input_frame.transpose(2, 0, 1)[np.newaxis, ...]

    current_time = time.time()
    new_face_timers = {}

    # Perform inference
    start_inference = time.time()
    detections = compiled_model([input_blob])[output_layer]
    inference_time = time.time() - start_inference

    for detection in detections[0, 0]:
        confidence = detection[2]
        if confidence > 0.5:
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
                        analysis = DeepFace.analyze(
                            face_roi, actions=['age', 'gender'], enforce_detection=False
                        )

                        if isinstance(analysis, list):
                            analysis = analysis[0]

                        age = analysis['age']
                        gender = analysis['dominant_gender'].capitalize()

                        text = f"{gender}, Age: {age}"
                        cv2.putText(
                            frame, text, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )

                        # Print analysis and inference time only when analysis succeeds
                        print(f"Inference time: {inference_time:.4f} seconds")
                        print([gender, age])

                    except Exception as e:
                        print(f"Error analyzing face: {e}")
            else:
                face_timers[face_id] = current_time

            new_face_timers[face_id] = face_timers.get(face_id, current_time)

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    face_timers = new_face_timers

    cv2.imshow("Pi Face Detection (OpenVINO)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
