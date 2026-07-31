"""
recognize.py — Etape 2 : reconnaissance en direct (OpenCV pur, sans dlib).

Ouvre la webcam et dessine un carre qui "scanne" ta tete :
    - VERT  si le visage correspond a ton profil (face_model.yml)
    - ROUGE sinon (visage inconnu)

Lance d'abord `python enroll.py` pour creer ton modele.

Le modele LBPH donne une "distance" (confidence) : plus elle est BASSE,
plus le visage ressemble au tien. On decide "c'est moi" si la distance est
sous le seuil THRESHOLD.

Commandes :
    +/-       ajuster le seuil (sensibilite)
    Q/ECHAP   quitter
"""

import sys
import time

import cv2
import numpy as np

MODEL_PATH = "face_model.yml"
CAM_INDEX = 0
FACE_SIZE = (200, 200)
# Distance LBPH sous laquelle on considere "c'est moi".
# Plus bas = plus strict. Typiquement entre 40 et 80 selon ta camera/lumiere.
THRESHOLD = 65.0

GREEN = (0, 200, 0)
RED = (0, 0, 255)

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if CASCADE.empty():
    sys.exit(
        "Fichier de detection de visage introuvable.\n"
        "Installe une version 4.x d'OpenCV :\n"
        "    pip install -r requirements.txt"
    )


def load_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(MODEL_PATH)
    except cv2.error:
        sys.exit("Modele introuvable ou illisible. Lance d'abord :  python enroll.py")
    return recognizer


def draw_scan_box(frame, box, color, label, phase):
    x, y, w, h = box
    top, left, bottom, right = y, x, y + h, x + w
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # coins renforces (style "scanner")
    c = 20
    for (px, py, dx, dy) in [
        (left, top, 1, 1), (right, top, -1, 1),
        (left, bottom, 1, -1), (right, bottom, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * c, py), color, 4)
        cv2.line(frame, (px, py), (px, py + dy * c), color, 4)

    # ligne de scan qui va et vient
    y_scan = int(top + (0.5 + 0.5 * np.sin(phase)) * h)
    cv2.line(frame, (left, y_scan), (right, y_scan), color, 1)

    # etiquette
    cv2.rectangle(frame, (left, top - 30), (right, top), color, cv2.FILLED)
    cv2.putText(frame, label, (left + 6, top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main():
    recognizer = load_model()
    threshold = THRESHOLD

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        sys.exit("Impossible d'ouvrir la webcam. Verifie les permissions camera de macOS.")

    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        phase = (time.time() - start) * 4

        for (x, y, w, h) in faces:
            crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            label_id, distance = recognizer.predict(crop)
            is_me = distance < threshold
            color = GREEN if is_me else RED
            text = f"MOI ({distance:.0f})" if is_me else f"INCONNU ({distance:.0f})"
            draw_scan_box(frame, (x, y, w, h), color, text, phase)

        cv2.putText(frame, f"Seuil: {threshold:.0f}  (+/- pour ajuster, Q pour quitter)",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Reconnaissance faciale", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            threshold = min(150.0, threshold + 2)
        elif key in (ord('-'), ord('_')):
            threshold = max(10.0, threshold - 2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
