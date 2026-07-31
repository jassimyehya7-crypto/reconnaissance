# reconnaissance

Prototype de reconnaissance faciale sur webcam (Mac).
Un carré « scanne » ta tête en direct : **vert** si ça te reconnaît, **rouge** sinon.

## Aperçu

- `enroll.py` — étape 1 : tu enregistres ton visage une fois (crée `known_face.pkl`).
- `recognize.py` — étape 2 : reconnaissance en direct avec le carré vert/rouge.

## Installation (macOS)

`face_recognition` s'appuie sur **dlib**, qui a besoin de `cmake` pour se compiler.

```bash
# 1. Outils de compilation
brew install cmake

# 2. Environnement Python isolé (recommandé)
python3 -m venv venv
source venv/bin/activate

# 3. Dépendances
pip install -r requirements.txt
```

> ⚠️ La première fois, `pip install` compile dlib : ça peut prendre plusieurs
> minutes. C'est normal.

## Utilisation

```bash
# Étape 1 — enregistrer ton visage
python enroll.py
#   C ou ESPACE  : capturer un échantillon (vise ~5, bien éclairé, face caméra)
#   S ou ENTRÉE  : sauvegarder
#   Q ou ÉCHAP   : quitter

# Étape 2 — reconnaissance en direct
python recognize.py
#   +/-      : ajuster la tolérance (sensibilité)
#   Q/ÉCHAP  : quitter
```

Au premier lancement, macOS demande l'autorisation d'accès à la caméra
(**Réglages Système → Confidentialité et sécurité → Caméra**).

## Réglage

La **tolérance** (distance) décide du seuil « c'est moi » :

- valeur **basse** (ex. 0.45) = plus strict, moins de faux positifs
- valeur **haute** (ex. 0.6) = plus permissif

Tu peux l'ajuster en direct avec `+` / `-` dans `recognize.py`, ou changer
la constante `TOLERANCE` dans le fichier.

## Limites (à savoir)

- Reconnaissance **2D** : vulnérable au *spoofing* (une photo de toi peut
  suffire à tromper le système). Pas d'équivalent Face ID sans caméra infrarouge.
- Prochaine étape possible : ajouter un **liveness check** (détection de
  clignement des yeux) pour distinguer un vrai visage d'une photo.
