import cv2
import time
from deepface import DeepFace

# Load OpenCV's pre-trained deep learning face detector
face_net = cv2.dnn.readNetFromCaffe(
    "/home/icepi/models/weights-prototxt.txt",
    "/home/icepi/models/res_ssd_300Dim.caffeModel"
)

cap = cv2.VideoCapture(0)

# Dictionary to store face timers
face_timers = {}

# Threshold time before processing a face (in seconds)
PROCESS_DELAY = 2.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    current_time = time.time()
    new_face_timers = {}

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x, y, x2, y2 = box.astype("int")

            # Generate a unique key for the detected face (based on position)
            face_id = (x // 10, y // 10, x2 // 10, y2 // 10)  # Reduce precision to group nearby detections

            # If face is seen for the first time, set initial time
            if face_id in face_timers:
                elapsed_time = current_time - face_timers[face_id]
                if elapsed_time >= PROCESS_DELAY:
                    # Extract face region for analysis
                    face_roi = frame[y:y2, x:x2]

                    try:
                        start_time = time.time()
                        analysis = DeepFace.analyze(face_roi, actions=['age', 'gender'], enforce_detection=False)
                        end_time = time.time()

                        age = analysis[0]['age']
                        gender = analysis[0]['dominant_gender'].capitalize()

                        text = f"{gender}, Age: {age}"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                        # **IMMEDIATELY PRINT OUTPUT**
                        print([gender, age])

                    except Exception as e:
                        print(f"Error processing face: {e}")
            else:
                # Store the new face with the current time
                face_timers[face_id] = current_time

            new_face_timers[face_id] = face_timers.get(face_id, current_time)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    # Update face_timers with only the currently detected faces
    face_timers = new_face_timers

    cv2.imshow("Optimized Multi-Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
