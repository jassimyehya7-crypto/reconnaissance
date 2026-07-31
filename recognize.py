"""
recognize.py — Etape 2 : reconnaissance en direct (YuNet + SFace).

Ouvre la webcam et dessine un carre qui "scanne" chaque visage :
    - VERT  si le visage correspond a ton profil (known_face.npz)
    - ROUGE sinon (visage inconnu)

On compare l'empreinte 128-D du visage a celles enregistrees, par
similarite cosinus : PLUS le score est HAUT, plus ca te ressemble.
"c'est moi" si le meilleur score depasse le seuil.

Lance d'abord `python enroll.py`.

Commandes :
    +/-       ajuster le seuil (sensibilite)
    Q/ECHAP   quitter
"""

import sys
import time

import cv2
import numpy as np

import face_utils

PROFILE_PATH = "known_face.npz"
CAM_INDEX = 0

GREEN = (0, 200, 0)
RED = (0, 0, 255)


def load_profile():
    try:
        data = np.load(PROFILE_PATH)
        return data["embeddings"].astype(np.float32)
    except FileNotFoundError:
        sys.exit("Profil introuvable. Lance d'abord :  python enroll.py")


def best_similarity(recognizer, feat, known):
    """Meilleure similarite cosinus entre le visage courant et les empreintes connues."""
    best = -1.0
    for k in known:
        s = face_utils.cosine(recognizer, feat, k.reshape(1, -1))
        if s > best:
            best = s
    return best


def draw_scan_box(frame, box, color, label, phase):
    x, y, w, h = box
    left, top, right, bottom = x, y, x + w, y + h
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    c = 20  # coins renforces
    for (px, py, dx, dy) in [
        (left, top, 1, 1), (right, top, -1, 1),
        (left, bottom, 1, -1), (right, bottom, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * c, py), color, 4)
        cv2.line(frame, (px, py), (px, py + dy * c), color, 4)

    y_scan = int(top + (0.5 + 0.5 * np.sin(phase)) * h)  # ligne de scan animee
    cv2.line(frame, (left, y_scan), (right, y_scan), color, 1)

    cv2.rectangle(frame, (left, top - 30), (right, top), color, cv2.FILLED)
    cv2.putText(frame, label, (left + 6, top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main():
    known = load_profile()
    detector = face_utils.make_detector()
    recognizer = face_utils.make_recognizer()
    threshold = face_utils.COSINE_THRESHOLD

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        sys.exit("Impossible d'ouvrir la webcam. Verifie les permissions camera de macOS.")

    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        clean = frame.copy()
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        phase = (time.time() - start) * 4

        if faces is not None:
            for face_row in faces:
                aligned = recognizer.alignCrop(clean, face_row)
                feat = recognizer.feature(aligned)
                score = best_similarity(recognizer, feat, known)
                is_me = score >= threshold
                color = GREEN if is_me else RED
                text = f"MOI ({score:.2f})" if is_me else f"INCONNU ({score:.2f})"
                box = face_row[:4].astype(int)
                draw_scan_box(frame, box, color, text, phase)

        cv2.putText(frame, f"Seuil: {threshold:.2f}  (+/- pour ajuster, Q pour quitter)",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Reconnaissance faciale", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            threshold = min(0.9, threshold + 0.02)
        elif key in (ord('-'), ord('_')):
            threshold = max(0.1, threshold - 0.02)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
