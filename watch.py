"""
watch.py — Surveillance : identite + posture + alerte.

Regle : ALERTE ROUGE seulement si (c'est MOI) ET (je suis ASSIS).
    - Toi debout        -> OK (vert)
    - Toi assis         -> ALERTE / INTERDIT (rouge clignotant)
    - Inconnu (papa...) -> rien, quelle que soit sa posture

Prerequis :
    python enroll.py             # ton visage       -> known_face.npz
    python calibrate_posture.py  # debout / assis   -> posture.npz

Commandes :
    +/-       ajuster le seuil d'identite
    Q/ECHAP   quitter
"""

import sys
import time

import cv2
import numpy as np

import face_utils

FACE_PROFILE = "known_face.npz"
POSTURE_PROFILE = "posture.npz"
CAM_INDEX = 0
SIT_STREAK = 6  # nb d'images consecutives "moi assis" avant de declencher l'alerte

GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)


def load_profiles():
    try:
        known = np.load(FACE_PROFILE)["embeddings"].astype(np.float32)
    except FileNotFoundError:
        sys.exit("Profil visage introuvable. Lance d'abord :  python enroll.py")
    try:
        p = np.load(POSTURE_PROFILE)
        stand_c = p["standing"].mean(axis=0)
        sit_c = p["sitting"].mean(axis=0)
    except FileNotFoundError:
        sys.exit("Profil posture introuvable. Lance d'abord :  python calibrate_posture.py")
    return known, stand_c, sit_c


def face_features(face_row, width, height):
    x, y, w, h = face_row[:4]
    return np.array([(y + h / 2.0) / height, h / height], dtype=np.float32)


def best_identity(recognizer, feat, known):
    best = -1.0
    for k in known:
        s = face_utils.cosine(recognizer, feat, k.reshape(1, -1))
        best = max(best, s)
    return best


def main():
    known, stand_c, sit_c = load_profiles()
    detector = face_utils.make_detector()
    recognizer = face_utils.make_recognizer()
    threshold = face_utils.COSINE_THRESHOLD

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        sys.exit("Impossible d'ouvrir la webcam. Verifie les permissions camera de macOS.")

    sit_streak = 0
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

        me_sitting_now = False

        if faces is not None:
            for face_row in faces:
                # --- identite ---
                aligned = recognizer.alignCrop(clean, face_row)
                feat = recognizer.feature(aligned)
                score = best_identity(recognizer, feat, known)
                is_me = score >= threshold

                # --- posture ---
                pf = face_features(face_row, w, h)
                d_stand = np.linalg.norm(pf - stand_c)
                d_sit = np.linalg.norm(pf - sit_c)
                sitting = d_sit < d_stand
                posture = "assis" if sitting else "debout"

                if is_me and sitting:
                    me_sitting_now = True

                # couleur du carre selon la combinaison
                if is_me and sitting:
                    color = RED
                elif is_me:
                    color = GREEN
                else:
                    color = ORANGE  # inconnu = on l'affiche mais pas d'alerte
                who = "MOI" if is_me else "INCONNU"
                label = f"{who} - {posture}"

                x, y, fw, fh = face_row[:4].astype(int)
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
                cv2.rectangle(frame, (x, y - 28), (x + fw, y), color, cv2.FILLED)
                cv2.putText(frame, label, (x + 5, y - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # lissage : il faut plusieurs images "moi assis" d'affilee
        sit_streak = sit_streak + 1 if me_sitting_now else 0
        alert = sit_streak >= SIT_STREAK

        if alert and int((time.time() - start) * 3) % 2 == 0:  # clignotement
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 12)
            cv2.rectangle(frame, (0, 0), (w, 60), RED, cv2.FILLED)
            cv2.putText(frame, "ALERTE - INTERDIT", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.putText(frame, f"Seuil identite: {threshold:.2f}  (+/- , Q=quitter)",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Surveillance", frame)
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
