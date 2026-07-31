"""
enroll.py — Etape 1 : enregistrer ton visage (OpenCV pur, sans dlib).

Ouvre la webcam, detecte ton visage, et enregistre plusieurs echantillons
en noir et blanc. On entraine un modele LBPH et on le sauvegarde dans
face_model.yml.

Commandes :
    ESPACE       -> demarrer / arreter la capture automatique
    C            -> capturer un seul echantillon
    R            -> tout remettre a zero
    S ou ENTREE  -> entrainer + sauvegarder + quitter
    Q ou ECHAP   -> quitter sans sauvegarder

Vise environ 30 echantillons, avec de petites variations (tourne un peu la
tete, souris, etc.). Plus il y a d'echantillons varies, mieux c'est.
"""

import sys

import cv2
import numpy as np

MODEL_PATH = "face_model.yml"
CAM_INDEX = 0
FACE_SIZE = (200, 200)   # taille normalisee des visages
TARGET_SAMPLES = 30

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_largest_face(gray):
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # plus grand visage
    return max(faces, key=lambda r: r[2] * r[3])


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        sys.exit("Impossible d'ouvrir la webcam. Verifie les permissions camera de macOS.")

    samples = []
    recording = False
    frame_count = 0
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # effet miroir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detect_largest_face(gray)
        frame_count += 1

        crop = None
        if face is not None:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
            crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)

            # capture auto : un echantillon tous les 3 frames pendant l'enregistrement
            if recording and frame_count % 3 == 0 and len(samples) < TARGET_SAMPLES:
                samples.append(crop)

        status = "ENREGISTREMENT..." if recording else "En pause (ESPACE pour lancer)"
        color = (0, 0, 255) if recording else (255, 255, 255)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Echantillons: {len(samples)}/{TARGET_SAMPLES}",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Enrolement - ESPACE: capture auto | S: sauver | Q: quitter", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            recording = not recording
        elif key in (ord('c'), ord('C')):
            if crop is not None:
                samples.append(crop)
                print(f"  Capture ({len(samples)}).")
        elif key in (ord('r'), ord('R')):
            samples.clear()
            recording = False
            print("  Remis a zero.")
        elif key in (ord('s'), ord('S'), 13):
            break
        elif key in (ord('q'), ord('Q'), 27):
            print("Abandon, rien n'a ete sauvegarde.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < 5:
        sys.exit(f"Seulement {len(samples)} echantillon(s) : c'est trop peu. Relance et vise ~30.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = np.zeros(len(samples), dtype=np.int32)  # une seule personne = label 0
    recognizer.train(samples, labels)
    recognizer.write(MODEL_PATH)
    print(f"\nModele sauvegarde dans {MODEL_PATH} ({len(samples)} echantillons).")
    print("Tu peux maintenant lancer :  python recognize.py")


if __name__ == "__main__":
    main()
