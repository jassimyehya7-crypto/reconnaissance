"""
face_utils.py — outils partages par enroll.py et recognize.py.

Detection : YuNet (cv2.FaceDetectorYN)
Reconnaissance : SFace (cv2.FaceRecognizerSF) -> empreinte 128-D + similarite cosinus

Les 2 modeles ONNX sont telecharges automatiquement dans le dossier models/
la premiere fois.
"""

import os
import sys
import urllib.request

import cv2

MODELS_DIR = "models"

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/" + YUNET_FILE
)
SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/" + SFACE_FILE
)

# tailles minimales attendues (garde-fou si le telechargement echoue)
YUNET_MIN_BYTES = 100_000
SFACE_MIN_BYTES = 1_000_000

# Seuil de similarite cosinus recommande par OpenCV : au-dessus = meme personne.
COSINE_THRESHOLD = 0.363


def _download(url, dest, min_bytes):
    print(f"Telechargement de {os.path.basename(dest)} (une seule fois)...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Echec du telechargement ({e}).\nVerifie ta connexion internet puis relance.")
    if os.path.getsize(dest) < min_bytes:
        os.remove(dest)
        sys.exit("Fichier telecharge trop petit (probablement corrompu). Relance.")


def _ensure(file_name, url, min_bytes):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(path):
        _download(url, path, min_bytes)
    return path


def make_detector(size=(320, 320)):
    path = _ensure(YUNET_FILE, YUNET_URL, YUNET_MIN_BYTES)
    # score_threshold=0.8 : ne garde que les detections surs -> pas de doubles carres
    return cv2.FaceDetectorYN.create(path, "", size, 0.8, 0.3, 5000)


def make_recognizer():
    path = _ensure(SFACE_FILE, SFACE_URL, SFACE_MIN_BYTES)
    return cv2.FaceRecognizerSF.create(path, "")


def cosine(recognizer, feat_a, feat_b):
    """Similarite cosinus entre deux empreintes (plus haut = plus ressemblant)."""
    return recognizer.match(feat_a, feat_b, cv2.FaceRecognizerSF_FR_COSINE)
