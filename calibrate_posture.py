"""
calibrate_posture.py — Etape "posture" : apprendre a distinguer DEBOUT / ASSIS.

Ta webcam ne voit que le haut du corps : on se base donc sur la position et la
taille de ton visage dans l'image, qui changent quand tu t'assois (tete plus
basse / plus proche). Tu enregistres quelques exemples de chaque posture, sur
TA webcam, dans TON installation.

Deroule :
    1. Mets-toi DEBOUT, appuie sur D, bouge un peu -> ~40 exemples.
    2. ASSIEDS-toi, appuie sur A, bouge un peu -> ~40 exemples.
    3. Appuie sur S pour sauvegarder (cree posture.npz).

Commandes :
    D  -> enregistrer la posture DEBOUT (bascule marche/arret)
    A  -> enregistrer la posture ASSIS  (bascule marche/arret)
    R  -> tout remettre a zero
    S  -> sauvegarder + quitter
    Q  -> quitter sans sauvegarder
"""

import sys

import cv2
import numpy as np

import face_utils

OUT_PATH = "posture.npz"
CAM_INDEX = 0
TARGET_EACH = 40


def face_features(face_row, width, height):
    """Position/taille du visage, normalisees par la taille de l'image."""
    x, y, w, h = face_row[:4]
    y_center = (y + h / 2.0) / height   # 0 = haut de l'image, 1 = bas
    size = h / height                    # grand = proche de la camera
    return [y_center, size]


def main():
    detector = face_utils.make_detector()
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        sys.exit("Impossible d'ouvrir la webcam. Verifie les permissions camera de macOS.")

    standing, sitting = [], []
    mode = None  # None, "debout" ou "assis"
    frame_count = 0
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        frame_count += 1

        if faces is not None and len(faces) > 0:
            face_row = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = face_row[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 200, 255), 2)
            if mode and frame_count % 2 == 0:
                feats = face_features(face_row, w, h)
                (standing if mode == "debout" else sitting).append(feats)

        label = {None: "En pause", "debout": "ENREGISTRE: DEBOUT", "assis": "ENREGISTRE: ASSIS"}[mode]
        color = {None: (255, 255, 255), "debout": (0, 200, 0), "assis": (0, 0, 255)}[mode]
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Debout: {len(standing)}/{TARGET_EACH}   Assis: {len(sitting)}/{TARGET_EACH}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "D=debout  A=assis  S=sauver  Q=quitter",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Calibrage posture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('d'), ord('D')):
            mode = None if mode == "debout" else "debout"
        elif key in (ord('a'), ord('A')):
            mode = None if mode == "assis" else "assis"
        elif key in (ord('r'), ord('R')):
            standing.clear(); sitting.clear(); mode = None
            print("  Remis a zero.")
        elif key in (ord('s'), ord('S'), 13):
            break
        elif key in (ord('q'), ord('Q'), 27):
            print("Abandon, rien n'a ete sauvegarde.")
            cap.release(); cv2.destroyAllWindows(); return

    cap.release()
    cv2.destroyAllWindows()

    if len(standing) < 5 or len(sitting) < 5:
        sys.exit("Pas assez d'exemples (vise ~40 debout ET ~40 assis). Relance.")

    np.savez(
        OUT_PATH,
        standing=np.array(standing, dtype=np.float32),
        sitting=np.array(sitting, dtype=np.float32),
    )
    print(f"\nPosture sauvegardee dans {OUT_PATH} "
          f"(debout: {len(standing)}, assis: {len(sitting)}).")
    print("Tu peux maintenant lancer :  python watch.py")


if __name__ == "__main__":
    main()
