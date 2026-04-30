"""
SafeGo AI - Flask Backend
Handles citizen registration, face storage, and alert coordination.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import base64
import uuid
import io
from datetime import datetime
import face_recognition
import numpy as np
import pickle
from PIL import Image as PILImage

app = Flask(__name__)
CORS(app)

DB_PATH   = "safego.db"
FACES_DIR = "registered_faces"
os.makedirs(FACES_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# DATABASE SETUP
# -----------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS citizens (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT,
            email TEXT NOT NULL,
            emergency_contact_name TEXT,
            emergency_contact_email TEXT NOT NULL,
            emergency_contact_phone TEXT,
            address TEXT,
            face_image_path TEXT,
            face_encoding BLOB,
            registered_at TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            citizen_id TEXT,
            citizen_name TEXT,
            camera_location TEXT,
            screenshot_path TEXT,
            alert_sent_to TEXT,
            alert_time TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()


# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_all_face_encodings():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, face_encoding FROM citizens WHERE face_encoding IS NOT NULL"
    ).fetchall()
    conn.close()
    encodings = []
    for row in rows:
        if row['face_encoding']:
            encoding = pickle.loads(row['face_encoding'])
            encodings.append({'id': row['id'], 'name': row['name'], 'encoding': encoding})
    return encodings


def b64_to_rgb_array(face_b64: str) -> np.ndarray:
    """
    Decode base64 image string -> clean uint8 RGB numpy array via PIL.
    Always returns (H, W, 3) uint8 C-contiguous — what dlib needs.
    """
    if ',' in face_b64:
        face_b64 = face_b64.split(',')[1]
    img_bytes = base64.b64decode(face_b64)
    pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.ascontiguousarray(np.array(pil, dtype=np.uint8))
    return arr


# -----------------------------------------------------------------------
# API ROUTES
# -----------------------------------------------------------------------

@app.route('/api/register', methods=['POST'])
def register_citizen():
    image_path = None
    try:
        data = request.json
        required = ['name', 'email', 'emergency_contact_email', 'face_image_base64']
        for field in required:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400

        # Decode image
        image = b64_to_rgb_array(data['face_image_base64'])
        print(f"[Register] shape={image.shape} dtype={image.dtype} C={image.flags['C_CONTIGUOUS']}")

        # Detect faces
        face_locations = face_recognition.face_locations(
            image, number_of_times_to_upsample=1, model="hog"
        )
        if not face_locations:
            print("[Register] upsample=1 found nothing, trying upsample=2...")
            face_locations = face_recognition.face_locations(
                image, number_of_times_to_upsample=2, model="hog"
            )
        if not face_locations:
            return jsonify({
                'success': False,
                'error': 'No face detected. Please use a clear, well-lit frontal face photo.'
            }), 400

        # Encode face
        face_encs = face_recognition.face_encodings(image, face_locations, num_jitters=1)
        if not face_encs:
            return jsonify({
                'success': False,
                'error': 'Could not encode face. Please try a different photo.'
            }), 400

        face_encoding_blob = pickle.dumps(face_encs[0])

        # Save image to disk
        citizen_id = str(uuid.uuid4())
        image_path = os.path.join(FACES_DIR, f"{citizen_id}.jpg")
        PILImage.fromarray(image).save(image_path, "JPEG", quality=95)

        # Save to database
        conn = get_db()
        conn.execute('''
            INSERT INTO citizens
            (id, name, phone, email, emergency_contact_name, emergency_contact_email,
             emergency_contact_phone, address, face_image_path, face_encoding, registered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            citizen_id,
            data['name'],
            data.get('phone', ''),
            data['email'],
            data.get('emergency_contact_name', ''),
            data['emergency_contact_email'],
            data.get('emergency_contact_phone', ''),
            data.get('address', ''),
            image_path,
            face_encoding_blob,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'message': f'Successfully registered {data["name"]}! You are now protected by SafeGo AI.',
            'citizen_id': citizen_id
        })

    except Exception as e:
        import traceback
        print("REGISTER ERROR:", e)
        traceback.print_exc()
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/identify', methods=['POST'])
def identify_face():
    try:
        data     = request.json
        face_b64 = data.get('face_image_base64', '')

        unknown_image = b64_to_rgb_array(face_b64)

        # Resize if too large
        h, w = unknown_image.shape[:2]
        if w > 1280:
            scale     = 1280 / w
            pil_small = PILImage.fromarray(unknown_image).resize(
                (int(w * scale), int(h * scale)), PILImage.LANCZOS
            )
            unknown_image = np.ascontiguousarray(np.array(pil_small, dtype=np.uint8))

        # Detect faces
        unknown_locations = face_recognition.face_locations(unknown_image, model="hog")
        if not unknown_locations:
            print("[Identify] upsample=1 failed, trying upsample=2...")
            unknown_locations = face_recognition.face_locations(
                unknown_image, number_of_times_to_upsample=2, model="hog"
            )
        if not unknown_locations:
            return jsonify({
                'identified': False,
                'reason': 'No face detected — ensure good lighting and face the camera directly'
            })

        print(f"[Identify] Faces found: {len(unknown_locations)}")

        # Encode
        unknown_encodings = face_recognition.face_encodings(
            unknown_image, unknown_locations, num_jitters=3
        )

        all_known = load_all_face_encodings()
        if not all_known:
            return jsonify({'identified': False, 'reason': 'No registered users in database'})

        known_encodings = [p['encoding'] for p in all_known]
        best_distance   = 1.0
        best_match      = None

        for unknown_enc in unknown_encodings:
            distances = face_recognition.face_distance(known_encodings, unknown_enc)
            best_idx  = int(np.argmin(distances))
            dist      = float(distances[best_idx])
            print(f"[Identify] Best distance: {dist:.4f}")
            if dist < best_distance:
                best_distance = dist
                best_match    = all_known[best_idx]

        TOLERANCE = 0.65
        if best_match and best_distance < TOLERANCE:
            conn    = get_db()
            citizen = conn.execute(
                "SELECT * FROM citizens WHERE id = ?", (best_match['id'],)
            ).fetchone()
            conn.close()
            return jsonify({
                'identified':              True,
                'citizen_id':              citizen['id'],
                'name':                    citizen['name'],
                'email':                   citizen['email'],
                'phone':                   citizen['phone'],
                'emergency_contact_name':  citizen['emergency_contact_name'],
                'emergency_contact_email': citizen['emergency_contact_email'],
                'emergency_contact_phone': citizen['emergency_contact_phone'],
                'confidence':              round(1.0 - best_distance, 3),
                'distance':                round(best_distance, 4)
            })

        return jsonify({
            'identified':    False,
            'reason':        'No match found',
            'best_distance': round(best_distance, 4)
        })

    except Exception as e:
        import traceback
        print(f"[Identify ERROR]: {e}")
        traceback.print_exc()
        return jsonify({'identified': False, 'error': str(e)}), 500


@app.route('/api/log_alert', methods=['POST'])
def log_alert():
    try:
        data     = request.json
        alert_id = str(uuid.uuid4())
        conn     = get_db()
        conn.execute('''
            INSERT INTO alerts
            (id, citizen_id, citizen_name, camera_location, screenshot_path, alert_sent_to, alert_time, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert_id,
            data.get('citizen_id',      'UNKNOWN'),
            data.get('citizen_name',    'Unregistered Person'),
            data.get('camera_location', 'Bengaluru, Karnataka, India'),
            data.get('screenshot_path', ''),
            data.get('alert_sent_to',   ''),
            datetime.now().isoformat(),
            'SENT'
        ))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'alert_id': alert_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn           = get_db()
    total_citizens = conn.execute("SELECT COUNT(*) FROM citizens").fetchone()[0]
    total_alerts   = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    recent_alerts  = conn.execute(
        "SELECT * FROM alerts ORDER BY alert_time DESC LIMIT 5"
    ).fetchall()
    conn.close()
    return jsonify({
        'total_registered': total_citizens,
        'total_alerts':     total_alerts,
        'recent_alerts':    [dict(a) for a in recent_alerts]
    })


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


if __name__ == '__main__':
    print("SafeGo AI Backend running on http://localhost:5000")
    app.run(debug=True, port=5000)