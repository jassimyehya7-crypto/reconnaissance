"""
enroll.py — Étape 1 : enregistrer ton visage.

Ouvre la webcam, affiche ton flux, et te laisse capturer quelques photos
de ton visage. On calcule un "encodage" facial (empreinte 128-D) pour
chaque capture, on fait la moyenne, et on sauvegarde le résultat dans
known_face.pkl.

Commandes :
    C ou ESPACE  -> capturer un échantillon (vise ~5, bien éclairé, face caméra)
    R            -> tout remettre à zéro
    S ou ENTRÉE  -> sauvegarder et quitter
    Q ou ÉCHAP   -> quitter sans sauvegarder
"""

import pickle
import sys

import cv2
import face_recognition
import numpy as np

PROFILE_PATH = "known_face.pkl"
CAM_INDEX = 0            # 0 = webcam par défaut (FaceTime sur Mac)
DETECT_SCALE = 0.25      # on détecte sur une image réduite = plus rapide
TARGET_SAMPLES = 5       # nombre d'échantillons conseillé


def find_largest_face(rgb_small):
    """Retourne (top, right, bottom, left) du plus grand visage, ou None."""
    locations = face_recognition.face_locations(rgb_small)
    if not locations:
        return None
    # plus grand visage = surface la plus grande
    return max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        sys.exit("Impossible d'ouvrir la webcam. Vérifie les permissions caméra de macOS.")

    encodings = []
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # effet miroir, plus naturel
        small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        box = find_largest_face(rgb_small)
        if box is not None:
            top, right, bottom, left = [int(v / DETECT_SCALE) for v in box]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 255), 2)
            cv2.putText(frame, "Visage detecte - appuie sur C pour capturer",
                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        else:
            cv2.putText(frame, "Aucun visage detecte",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Echantillons: {len(encodings)}/{TARGET_SAMPLES}",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Enrolement - C: capturer | S: sauver | Q: quitter", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('c'), ord('C'), ord(' ')):
            if box is None:
                print("  Pas de visage à capturer.")
                continue
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            full_box = tuple(int(v / DETECT_SCALE) for v in box)
            found = face_recognition.face_encodings(rgb_full, [full_box])
            if found:
                encodings.append(found[0])
                print(f"  Capturé ({len(encodings)}).")

        elif key in (ord('r'), ord('R')):
            encodings.clear()
            print("  Remis à zéro.")

        elif key in (ord('s'), ord('S'), 13):  # 13 = Entrée
            break

        elif key in (ord('q'), ord('Q'), 27):  # 27 = Échap
            print("Abandon, rien n'a été sauvegardé.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if not encodings:
        sys.exit("Aucun échantillon capturé — rien à sauvegarder.")

    mean_encoding = np.mean(encodings, axis=0)
    with open(PROFILE_PATH, "wb") as f:
        pickle.dump({"encoding": mean_encoding, "samples": len(encodings)}, f)
    print(f"\nProfil sauvegardé dans {PROFILE_PATH} ({len(encodings)} échantillons).")
    print("Tu peux maintenant lancer :  python recognize.py")


if __name__ == "__main__":
    main()
