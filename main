import cv2
import mediapipe as mp
import time
from playsound import playsound

from pyflipper.pyflipper import PyFlipper

playsound("stun.mov")

#Local serial port
flipper = PyFlipper(com="/dev/tty.usbmodemflip_R4nc31")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Fingertip landmarks
FINGERTIPS = [4, 8, 12, 16, 20]
# Forehead area (some points at the top of the face)
FOREHEAD = [10, 21, 54, 103, 151, 338, 300, 332]

cap = cv2.VideoCapture(0)

TIME_DELAY = 1

# Initialize time of last alert to 10 seconds in the past
last_alert_time = time.time() - TIME_DELAY


def flipperShock():
    flipper.input.send("ok", "press")
    time.sleep(.1)
    flipper.input.send("ok", "release")

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        



        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results_hands = hands.process(image)
        results_face = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        if results_hands.multi_hand_landmarks and results_face.multi_face_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for idx in FINGERTIPS:
                    fingertip = hand_landmarks.landmark[idx]
                    fingertip_x = fingertip.x
                    fingertip_y = fingertip.y

                    # Check if fingertips are near the face (considering nose tip) or the forehead
                    for idf in FOREHEAD:
                        forehead_point = results_face.multi_face_landmarks[0].landmark[idf]
                        forehead_x = forehead_point.x
                        forehead_y = forehead_point.y

                        if (abs(fingertip_x - forehead_x) < 0.1 and abs(fingertip_y - forehead_y) < 0.1):
                            current_time = time.time()
                            if current_time - last_alert_time >= TIME_DELAY:  # More than 10 seconds since last alert
                                print("You are touching your hair!")
                                cv2.putText(image, "HAND DETECTED", (100, 600), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 2)  # Draw "Hi" at the locatio
                                playsound("stun.mov")
                                flipperShock()
                                last_alert_time = current_time
                            break

                    face_center = results_face.multi_face_landmarks[0].landmark[1]  # landmark 1 generally refers to the tip of the nose
                    face_x = face_center.x
                    face_y = face_center.y

                    if abs(fingertip_x - face_x) < 0.1 and abs(fingertip_y - face_y) < 0.1:
                        current_time = time.time()
                        if current_time - last_alert_time >= TIME_DELAY:  # More than 10 seconds since last alert
                            print("You are touching your face!")
                            cv2.putText(image, "HAND DETECTED", (100, 160), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 2)  # Draw "Hi" at the locatio
                            playsound("stun.mov")
                            flipperShock()
                            last_alert_time = current_time
                        break

        cv2.imshow('MediaPipe Hands and Face', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

