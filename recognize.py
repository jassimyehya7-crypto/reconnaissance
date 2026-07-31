"""
recognize.py — Étape 2 : reconnaissance en direct.

Ouvre la webcam et dessine un carré qui "scanne" ta tête :
    - VERT  si le visage correspond à ton profil (known_face.pkl)
    - ROUGE sinon (visage inconnu)

Lance d'abord `python enroll.py` pour créer ton profil.

Commandes :
    +/-   ajuster la tolérance (sensibilité)
    Q/ÉCHAP  quitter
"""

import pickle
import sys
import time

import cv2
import face_recognition
import numpy as np

PROFILE_PATH = "known_face.pkl"
CAM_INDEX = 0
DETECT_SCALE = 0.25
# Distance en dessous de laquelle on considère "c'est moi".
# Plus bas = plus strict. 0.6 est la valeur par défaut de face_recognition.
TOLERANCE = 0.5

GREEN = (0, 200, 0)
RED = (0, 0, 255)


def load_profile():
    try:
        with open(PROFILE_PATH, "rb") as f:
            data = pickle.load(f)
        return data["encoding"]
    except FileNotFoundError:
        sys.exit("Profil introuvable. Lance d'abord :  python enroll.py")


def draw_scan_box(frame, box, color, label, phase):
    """Dessine le carré + une ligne de scan animée + le label."""
    top, right, bottom, left = box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # coins renforcés (style "scanner")
    c = 20
    for (x, y, dx, dy) in [
        (left, top, 1, 1), (right, top, -1, 1),
        (left, bottom, 1, -1), (right, bottom, -1, -1),
    ]:
        cv2.line(frame, (x, y), (x + dx * c, y), color, 4)
        cv2.line(frame, (x, y), (x, y + dy * c), color, 4)

    # ligne de scan qui va et vient
    span = bottom - top
    y_scan = int(top + (0.5 + 0.5 * np.sin(phase)) * span)
    cv2.line(frame, (left, y_scan), (right, y_scan), color, 1)

    # étiquette
    cv2.rectangle(frame, (left, top - 30), (right, top), color, cv2.FILLED)
    cv2.putText(frame, label, (left + 6, top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main():
    known = load_profile()
    tolerance = TOLERANCE

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        sys.exit("Impossible d'ouvrir la webcam. Vérifie les permissions caméra de macOS.")

    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)
        phase = (time.time() - start) * 4  # vitesse de la ligne de scan

        for loc, enc in zip(locations, encodings):
            distance = np.linalg.norm(known - enc)
            is_me = distance < tolerance
            color = GREEN if is_me else RED
            label = f"MOI ({distance:.2f})" if is_me else f"INCONNU ({distance:.2f})"
            box = tuple(int(v / DETECT_SCALE) for v in loc)
            draw_scan_box(frame, box, color, label, phase)

        cv2.putText(frame, f"Tolerance: {tolerance:.2f}  (+/- pour ajuster, Q pour quitter)",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Reconnaissance faciale", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            tolerance = min(1.0, tolerance + 0.02)
        elif key in (ord('-'), ord('_')):
            tolerance = max(0.1, tolerance - 0.02)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
