import cv2
from deepface import DeepFace

# Load OpenCV's pre-trained deep learning face detector
face_net = cv2.dnn.readNetFromCaffe(
    "/home/icepi/models/weights-prototxt.txt",
    "/home/icepi/models/res_ssd_300Dim.caffeModel"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x, y, x2, y2 = box.astype("int")

            face_roi = frame[y:y2, x:x2]

            try:
                analysis = DeepFace.analyze(face_roi, actions=['age', 'gender'], enforce_detection=False)

                age = analysis[0]['age']
                gender = analysis[0]['dominant_gender'].capitalize()

                text = f"{gender}, Age: {age}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Print output immediately
                print([gender, age])

            except Exception as e:
                print(f"Error processing face: {e}")

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
