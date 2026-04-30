"""
SafeGo AI — Detection Script
Street camera agent: detects Signal for Help gesture, identifies face via
the SafeGo AI backend database, and dispatches alerts accordingly.

Usage:
    python detect.py --camera 0 --location "MG Road & Brigade Road, Bengaluru" --police "police_station@bengaluru.gov.in"

Dependencies:
    pip install opencv-python mediapipe requests face_recognition numpy
"""

import cv2
import mediapipe as mp
import time
import math
import requests
import smtplib
from email.message import EmailMessage
import datetime
import os
import numpy as np
import argparse
import base64

# ============================================================
# ⚙️ CONFIGURATION — Edit these or pass via CLI arguments
# ============================================================

SAFEGO_BACKEND_URL = "http://localhost:5000/api"  # Flask backend URL

# Email sender (the system account — use App Password for Gmail)
SENDER_EMAIL    = "ashiprojj@gmail.com"
SENDER_PASSWORD = "yeqewqxbxgyvzhfk"  # Gmail App Password (not real password)
# Nearest police station contact (set per-camera deployment)
DEFAULT_POLICE_EMAIL   = "ashivrma1289@gmail.com"
DEFAULT_POLICE_CONTACT = "100"  # National police number (India)

# Detection sensitivity
ALERT_COOLDOWN_SECONDS       = 300   # 5 min between repeat alerts for same event
CONSECUTIVE_DETECTION_FRAMES = 45    # ~1.5 seconds at 30fps

# ============================================================
# CLI ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description="SafeGo AI — Roadside Distress Detection")
parser.add_argument("--camera",   type=int,   default=0,      help="Camera index (default: 0)")
parser.add_argument("--location", type=str,   default="Bengaluru, Karnataka, India", help="Physical location of this camera")
parser.add_argument("--police",   type=str,   default=DEFAULT_POLICE_EMAIL,    help="Police station email for alerts")
args = parser.parse_args()

CAMERA_LOCATION  = args.location
POLICE_EMAIL     = args.police

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands      = mp.solutions.hands
mp_drawing    = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================================================
# FACE DETECTION SETUP (OpenCV built-in — no extra install)
# ============================================================

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ============================================================
# 🗺️ GEOLOCATION
# ============================================================

def get_approximate_location():
    try:
        r = requests.get("https://ipinfo.io/json", timeout=5)
        d = r.json()

        loc = d.get("loc", None)  # "lat,lon"
        city = d.get("city", "Bengaluru")
        region = d.get("region", "Karnataka")
        country = d.get("country", "India")

        if loc:
            lat, lng = loc.split(",")
            map_url = f"https://maps.google.com/?q={lat},{lng}"
        else:
            map_url = None

        details = f"{CAMERA_LOCATION} - approx. {city},{region},{country}"

        return details, map_url

    except Exception as e:
        print(f"[GeoLocation] Error: {e}")
        return f"{CAMERA_LOCATION}", None


# ============================================================
# 📧 ALERT FUNCTIONS
# ============================================================

def send_ntfy_push(title: str, body: str, map_url: str = None):
    """Send an urgent push notification via ntfy.sh."""
    try:
        clean_title = str(title).strip().replace("—", "-").encode("ascii", "ignore").decode()
        clean_body = str(body).strip()
        headers = {
            "Title": clean_title,
            "Priority": "5",
            #"Tags": "warning,sos",
            "Content-Type": "text/plain; charset=utf-8",
            }
        
        #if map_url and map_url.startswith("http"):
            #headers["Actions"] = f"view,Open Map,{map_url}"

        r = requests.post(
            "https://ntfy.sh/safego-alerts",
            data=clean_body.encode("utf-8"),
            headers=headers,
            timeout=10
        )
        print(f"[NTFY] {'Sent' if r.status_code == 200 else f'Failed ({r.status_code})'}")
    except Exception as e:
        print(f"[NTFY] Error: {e}")


def send_email(to_email: str, subject: str, body: str, image_path: str = None):
    """Send an email alert with optional screenshot attachment."""
    try:
        msg            = EmailMessage()
        msg["Subject"] = subject
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = to_email
        msg.set_content(body)

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="image", subtype="jpeg",
                                   filename=os.path.basename(image_path))

        import ssl
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.ehlo()
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)

        print(f"[Email] Sent to {to_email}")
    except Exception as e:
        print(f"[Email] Failed to send to {to_email}: {e}")


def frame_to_base64(frame) -> str:
    """Encode an OpenCV frame to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def identify_person(frame) -> dict:
    try:
        b64 = frame_to_base64(frame)
        r = requests.post(
            f"{SAFEGO_BACKEND_URL}/identify",
            json={"face_image_base64": b64},
            timeout=10
        )
        return r.json()
    except Exception as e:
        print(f"[Identify] Backend unreachable: {e}")
        return {"identified": False, "reason": "Backend error"}


def log_alert_to_backend(citizen_id, citizen_name, screenshot_path, alert_sent_to):
    """Log the alert event to the SafeGo backend database."""
    try:
        requests.post(
            f"{SAFEGO_BACKEND_URL}/log_alert",
            json={
                "citizen_id":      citizen_id,
                "citizen_name":    citizen_name,
                "camera_location": CAMERA_LOCATION,
                "screenshot_path": screenshot_path,
                "alert_sent_to":   alert_sent_to,
            },
            timeout=5
        )
    except Exception:
        pass  # Non-critical


def dispatch_alerts(person_data: dict, screenshot_path: str, timestamp: str):
    """
    Route alerts based on whether person is registered or not.
    - Registered person → Emergency contact + Police
    - Unregistered       → Police only
    """
    location_str, map_url = get_approximate_location()
    name = person_data.get("name", "Unregistered Person")
    is_registered = person_data.get("identified", False)

    alert_recipients = []

    # ── POLICE alert (always sent) ──────────────────────────────────────────
    police_subject = f"SAFEGO AI ALERT - Distress Signal at {CAMERA_LOCATION}"
    police_body = (
        f"URGENT: A distress signal (Signal for Help gesture) has been detected.\n\n"
        f"Identity: {'REGISTERED - ' + name if is_registered else 'UNREGISTERED PERSON'}\n"
        f"Camera Location: {CAMERA_LOCATION}\n"
        f"GPS Approx: {location_str}\n"
        f"Map: {map_url or 'N/A'}\n"
        f"Time: {timestamp}\n\n"
        f"Please dispatch the nearest unit immediately.\n"
        f"A screenshot of the person is attached."
    )
    send_email(POLICE_EMAIL, police_subject, police_body, screenshot_path)
    alert_recipients.append(f"police:{POLICE_EMAIL}")

    # ── NTFY push (police dashboard) ────────────────────────────────────────
    send_ntfy_push(
        title=f"Distress Signal - {name}",
        body=f"{CAMERA_LOCATION}. {'Registered citizen: ' + name if is_registered else 'Unregistered person.'}. Time: {timestamp}. {location_str}",
        map_url=map_url
    )

    # ── EMERGENCY CONTACT (registered persons only) ─────────────────────────
    if is_registered:
        ec_email = person_data.get("emergency_contact_email")
        ec_name  = person_data.get("emergency_contact_name", "Emergency Contact")
        phone    = person_data.get("phone", "N/A")

        if ec_email:
            ec_subject = f" URGENT: {name} may need help!"
            ec_body = (
                f"Dear {ec_name},\n\n"
                f"This is an automated URGENT alert from SafeGo AI.\n\n"
                f"{name} has shown the Signal for Help gesture at a monitored location.\n\n"
                f" Location: {CAMERA_LOCATION}\n"
                f"Map: {map_url or 'N/A'}\n"
                f" Screenshot attached\n"
                f" Time: {timestamp}\n\n"
                f"Police have been notified. Their contact: {DEFAULT_POLICE_CONTACT}\n\n"
                f"Please try to contact {name} immediately at {phone}.\n\n"
                f"— SafeGo AI Safety Network"
            )
            send_email(ec_email, ec_subject, ec_body, screenshot_path)
            alert_recipients.append(f"emergency_contact:{ec_email}")

    # ── Log to backend ───────────────────────────────────────────────────────
    log_alert_to_backend(
        citizen_id     = person_data.get("citizen_id", "UNKNOWN"),
        citizen_name   = name,
        screenshot_path= screenshot_path,
        alert_sent_to  = ", ".join(alert_recipients)
    )

    print(f"\n{'='*60}")
    print(f"[ALERT DISPATCHED] {timestamp}")
    print(f"  Person   : {name} ({'Registered' if is_registered else 'Unregistered'})")
    print(f"  Location : {CAMERA_LOCATION}")
    print(f"  Sent to  : {', '.join(alert_recipients)}")
    print(f"{'='*60}\n")


# ============================================================
# GESTURE DETECTION
# ============================================================

def get_distance(p1, p2) -> float:
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)


def is_signal_for_help(hand_landmarks) -> bool:
    """
    Checks the 'Signal for Help' gesture:
      1. Thumb is tucked inward (tip near index MCP, relative to hand size)
      2. All four fingers are folded down over the thumb
    """
    lm = hand_landmarks.landmark
    THUMB_TIP  = mp_hands.HandLandmark.THUMB_TIP
    THUMB_MCP  = mp_hands.HandLandmark.THUMB_MCP
    INDEX_MCP  = mp_hands.HandLandmark.INDEX_FINGER_MCP
    WRIST      = mp_hands.HandLandmark.WRIST

    thumb_tip  = lm[THUMB_TIP]
    thumb_mcp  = lm[THUMB_MCP]
    index_mcp  = lm[INDEX_MCP]
    wrist      = lm[WRIST]

    hand_size  = get_distance(wrist, thumb_mcp)
    if hand_size < 1e-5:
        return False

    # Thumb folded inward check
    if get_distance(thumb_tip, index_mcp) / hand_size >= 0.35:
        return False

    # Fingers folded over check
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,  mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,   mp_hands.HandLandmark.PINKY_TIP]
    mcps = [mp_hands.HandLandmark.INDEX_FINGER_MCP,  mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_MCP,   mp_hands.HandLandmark.PINKY_MCP]

    folded = sum(
        1 for t, m in zip(tips, mcps)
        if get_distance(lm[t], wrist) / get_distance(lm[m], wrist) < 0.80
    )
    return folded == 4


# ============================================================
# DISPLAY HELPER — draw face box with label
# ============================================================

def draw_face_boxes(frame, gray, is_alert: bool, display_name: str):
    """
    Detect faces and draw:
      - GREEN box  + name/Unknown  → normal state
      - RED box    + name/Unknown  → help signal detected
    """
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    box_color   = (0, 0, 220) if is_alert else (0, 220, 0)   # red or green (BGR)
    label_bg    = (0, 0, 180) if is_alert else (0, 180, 0)
    FONT        = cv2.FONT_HERSHEY_SIMPLEX

    for (fx, fy, fw, fh) in faces:
        # Face rectangle
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)

        # Name label background + text
        label_text = display_name
        (tw, th), _ = cv2.getTextSize(label_text, FONT, 0.6, 2)
        label_y = fy + fh + 4
        cv2.rectangle(frame, (fx, label_y), (fx + tw + 8, label_y + th + 8), label_bg, -1)
        cv2.putText(frame, label_text, (fx + 4, label_y + th + 2), FONT, 0.6, (255, 255, 255), 2)


# ============================================================
# MAIN DETECTION LOOP
# ============================================================

def main():
    print(f"\n{'='*60}")
    print("  SafeGo AI — Distress Detection System")
    print(f"  Camera Location : {CAMERA_LOCATION}")
    print(f"  Police Email    : {POLICE_EMAIL}")
    print(f"  Backend URL     : {SAFEGO_BACKEND_URL}")
    print(f"{'='*60}\n")
    print("Press Q to quit.\n")

    # Camera open
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("FATAL: Could not open camera. Check index/permissions.")
        hands.close()
        return

    gesture_frames           = 0
    last_alert_time          = 0.0
    alert_sent_current_event = False
    last_identified_person   = {}

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # for face detection
        h, w      = frame.shape[:2]

        gesture_in_frame = False

        # ── Hand detection ────────────────────────────────────────────────
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,220,130), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,168,255), thickness=2))

                if is_signal_for_help(hl):
                    gesture_in_frame = True

        # ── Determine display name for face box ───────────────────────────
        if last_identified_person.get("identified"):
            display_name = last_identified_person.get("name", "Unknown")
        elif gesture_frames == 0:
            display_name = ""          # show nothing when just standing normally
        else:
            display_name = "Unknown"   # show Unknown only while actively matching

        # ── Draw face boxes (GREEN normal, RED on help signal) ─────────────
        is_alert_state = gesture_in_frame or gesture_frames > 0
        draw_face_boxes(frame, gray, is_alert=is_alert_state, display_name=display_name)

        # ── Status overlay ────────────────────────────────────────────────
        if gesture_in_frame:
            cv2.rectangle(frame, (0, 0), (w, 68), (0, 0, 180), -1)
            cv2.putText(frame, "HELP SIGNAL DETECTED!", (12, 44),
                        FONT, 0.9, (255, 255, 255), 2)

        # Progress bar
        if gesture_frames > 0:
            progress = min(gesture_frames / CONSECUTIVE_DETECTION_FRAMES, 1.0)
            bar_w    = int(w * progress)
            cv2.rectangle(frame, (0, h-8), (bar_w, h), (0, 220, 130), -1)
            cv2.putText(frame, f"Confirming: {int(progress*100)}%", (8, h-16),
                        FONT, 0.55, (0, 220, 130), 1)

        # Camera info overlay
        cv2.putText(frame, f" {CAMERA_LOCATION}", (8, h-30),
                    FONT, 0.45, (150, 150, 150), 1)

        # ── Temporal smoothing ───────────────────────────────────────────
        now = time.time()

        if gesture_in_frame:
            gesture_frames += 1
        else:
            gesture_frames           = 0
            alert_sent_current_event = False
            last_identified_person   = {}

        # ── Alert trigger ────────────────────────────────────────────────
        if (gesture_frames >= CONSECUTIVE_DETECTION_FRAMES and
                not alert_sent_current_event and
                (now - last_alert_time) > ALERT_COOLDOWN_SECONDS):

            print("[SafeGo] Gesture confirmed. Identifying person...")

            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 220), 4)
            cv2.putText(frame, "ALERT TRIGGERED - IDENTIFYING...", (12, 80),
                        FONT, 0.8, (255, 255, 100), 2)

            # Save screenshot
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot = f"alert_{ts}.jpg"
            cv2.imwrite(screenshot, frame)

            # Identify via backend
            person_data = identify_person(frame)
            last_identified_person = person_data

            # ── Update display immediately after identification ──
            cv2.imshow("SafeGo AI — Distress Monitor", frame)
            cv2.waitKey(1)
            # Dispatch alerts
            dispatch_alerts(
                person_data    = person_data,
                screenshot_path= screenshot,
                timestamp      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            alert_sent_current_event = True
            last_alert_time          = now

        # ── Identity overlay (below face box now — shown via draw_face_boxes) ──
        # Show matching status text only while confirming, before identity is known
        if not last_identified_person and gesture_frames > 10:
            cv2.putText(frame, "Matching face...", (12, 105),
                        FONT, 0.65, (0, 168, 255), 2)

        cv2.imshow("SafeGo AI — Distress Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n[SafeGo] Session ended.")


if __name__ == "__main__":
    main()