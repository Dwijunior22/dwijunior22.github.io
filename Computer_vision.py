import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Variables to keep track of gesture
click_start_time = None
click_threshold = 0.5  # Time in seconds to recognize a click

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordinates for the tip of the index finger
            x_index = hand_landmarks.landmark[8].x
            y_index = hand_landmarks.landmark[8].y

            # Convert the normalized coordinates to screen coordinates
            screen_x = pyautogui.size().width * x_index
            screen_y = pyautogui.size().height * y_index

            # Move the cursor to the index finger tip location
            pyautogui.moveTo(screen_x, screen_y)

            # Check for click (index finger held in a stable position)
            if click_start_time is None:
                click_start_time = time.time()
            elif time.time() - click_start_time > click_threshold:
                pyautogui.click()
                click_start_time = None
            else:
                click_start_time = time.time()

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
