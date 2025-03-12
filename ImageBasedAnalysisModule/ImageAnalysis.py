import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import mediapipe as mp

# Load Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load Trained Gender & Age Model
MODEL_PATH = "/home/icepi/models/gender_age_model.h5"
gender_age_model = load_model(MODEL_PATH, compile=False)

def preprocess_image(image):
    """ Preprocess the image for the gender & age classification model """
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize to [0,1]
    return image

def predict_gender_age(face_roi):
    """ Predict gender and age from the cropped face ROI """
    processed_image = preprocess_image(face_roi)

    # Disable progress bar by setting verbose=0
    predictions = gender_age_model.predict(processed_image, verbose=0)

    # Extract predictions as scalars
    gender_pred, age_pred = predictions[0][0], predictions[1][0]

    # Convert probability to gender
    gender = "Male" if gender_pred > 0.5 else "Female"

    # Adjust age scaling
    estimated_age = max(1, int(round(float(age_pred) * 6)))

    # Output results as a list
    result = [gender, estimated_age]
    print(f"✅ Detected: {result}")  # Debugging output

    return result


def estimate_distance(face_boxes, frame_width):
    """ Estimate the relative distance between detected faces """
    distances = []
    face_boxes = sorted(face_boxes, key=lambda box: box[0])  # Sort faces by x position

    for i in range(len(face_boxes) - 1):
        x1, _, w1, _ = face_boxes[i]
        x2, _, w2, _ = face_boxes[i + 1]

        face_size_ratio = (w1 + w2) / 2
        pixel_distance = abs(x2 - x1)

        # Simple heuristic: closer faces have larger bounding boxes
        normalized_distance = pixel_distance / (frame_width * face_size_ratio)
        distances.append(normalized_distance)

    return distances

def analyze_frame(frame):
    """ Analyze a single frame and return detected gender, age, and distance between faces """
    results_list = []
    face_boxes = []

    # Convert to RGB (for Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Ensure bounding box is within image bounds
            x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

            # Extract face ROI
            face_roi = frame[y:y + h, x:x + w]

            if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                gender, age = predict_gender_age(face_roi)
                results_list.append((gender, age, x, y, w, h))
                face_boxes.append((x, y, w, h))

    distances = estimate_distance(face_boxes, frame.shape[1])

    return results_list, distances

# Open Camera (Optional)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detected_faces, face_distances = analyze_frame(frame)

    # Draw Results on Frame
    for i, (gender, age, x, y, w, h) in enumerate(detected_faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{gender}, Age: {age}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display estimated distances between people
    if len(face_distances) > 0:
        for i, distance in enumerate(face_distances):
            text_position = (50, 50 + (i * 20))
            cv2.putText(frame, f"Distance {i+1}: {distance:.2f}", text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Mediapipe + Gender & Age Classification + Distance Estimation", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
