import cv2
import time
import threading
from pynput import keyboard
from plyer import notification

# ---------------- CONFIGURATION ---------------- #
INACTIVITY_LIMIT = 10     # seconds before alert
FACE_ABSENT_LIMIT = 10    # seconds before alert

# ------------------------------------------------ #
last_key_time = time.time()
last_face_time = time.time()


def send_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        timeout=5
    )


# ---------- KEYBOARD ACTIVITY MONITOR ---------- #
def on_press(key):
    global last_key_time
    last_key_time = time.time()


def keyboard_monitor():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ---------- FACE DETECTION MONITOR ---------- #
def face_monitor():
    global last_face_time

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            last_face_time = time.time()

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("SmartFocus - Focus Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- INACTIVITY CHECKER ---------- #
def inactivity_checker():
    global last_key_time, last_face_time

    while True:
        now = time.time()

        if now - last_key_time > INACTIVITY_LIMIT:
            send_notification("⚠ Focus Alert", "No keyboard activity detected! Stay focused 👀")
            last_key_time = time.time()

        if now - last_face_time > FACE_ABSENT_LIMIT:
            send_notification("⚠ Focus Alert", "Face not detected! Please stay focused 📸")
            last_face_time = time.time()

        time.sleep(5)


# ----------------- MAIN ----------------- #
if __name__ == "__main__":
    print("🔵 SmartFocus - Focus Monitoring System Started...")
    print("Press 'q' to quit camera window")

    t1 = threading.Thread(target=keyboard_monitor, daemon=True)
    t2 = threading.Thread(target=face_monitor, daemon=True)
    t3 = threading.Thread(target=inactivity_checker, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    t2.join()
