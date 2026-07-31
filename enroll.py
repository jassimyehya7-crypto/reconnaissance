"""
enroll.py — Etape 1 : enregistrer ton visage (YuNet + SFace).

Ouvre la webcam, detecte ton visage, et enregistre plusieurs empreintes
128-D. On les sauvegarde dans known_face.npz.

Commandes :
    ESPACE       -> demarrer / arreter la capture automatique
    C            -> capturer une seule empreinte
    R            -> tout remettre a zero
    S ou ENTREE  -> sauvegarder + quitter
    Q ou ECHAP   -> quitter sans sauvegarder

Vise environ 20 empreintes, avec de petites variations (tourne un peu la
tete, souris, avec/sans lunettes...).
"""

import sys

import cv2
import numpy as np

import face_utils

OUT_PATH = "known_face.npz"
CAM_INDEX = 0
TARGET_SAMPLES = 20


def main():
    detector = face_utils.make_detector()
    recognizer = face_utils.make_recognizer()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        sys.exit("Impossible d'ouvrir la webcam. Verifie les permissions camera de macOS.")

    embeddings = []
    recording = False
    frame_count = 0
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        clean = frame.copy()  # image sans dessins, pour aligner proprement
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        frame_count += 1

        face_row = None
        if faces is not None and len(faces) > 0:
            face_row = max(faces, key=lambda f: f[2] * f[3])  # plus grand visage
            x, y, fw, fh = face_row[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 200, 255), 2)

            if recording and frame_count % 3 == 0 and len(embeddings) < TARGET_SAMPLES:
                aligned = recognizer.alignCrop(clean, face_row)
                feat = recognizer.feature(aligned)
                embeddings.append(feat.flatten().copy())

        status = "ENREGISTREMENT..." if recording else "En pause (ESPACE pour lancer)"
        color = (0, 0, 255) if recording else (255, 255, 255)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Empreintes: {len(embeddings)}/{TARGET_SAMPLES}",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Enrolement - ESPACE: capture auto | S: sauver | Q: quitter", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            recording = not recording
        elif key in (ord('c'), ord('C')):
            if face_row is not None:
                aligned = recognizer.alignCrop(clean, face_row)
                embeddings.append(recognizer.feature(aligned).flatten().copy())
                print(f"  Capture ({len(embeddings)}).")
        elif key in (ord('r'), ord('R')):
            embeddings.clear()
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

    if len(embeddings) < 3:
        sys.exit(f"Seulement {len(embeddings)} empreinte(s) : trop peu. Relance et vise ~20.")

    np.savez(OUT_PATH, embeddings=np.array(embeddings, dtype=np.float32))
    print(f"\nProfil sauvegarde dans {OUT_PATH} ({len(embeddings)} empreintes).")
    print("Tu peux maintenant lancer :  python recognize.py")


if __name__ == "__main__":
    main()
