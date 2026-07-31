# reconnaissance

Prototype de reconnaissance faciale sur webcam (Mac), en **OpenCV pur**.
Un carré « scanne » ta tête en direct : **vert** si ça te reconnaît, **rouge** sinon.

Pas besoin de Homebrew, ni de cmake, ni de compiler quoi que ce soit.

## Aperçu

- `enroll.py` — étape 1 : tu enregistres ton visage une fois (crée `face_model.yml`).
- `recognize.py` — étape 2 : reconnaissance en direct avec le carré vert/rouge.

## Installation (macOS)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Utilisation

```bash
python enroll.py
```
- `ESPACE` : démarrer/arrêter la capture automatique (vise ~30 échantillons)
- `C` : capturer un seul échantillon
- `S` ou `ENTRÉE` : entraîner + sauvegarder
- `Q` ou `ÉCHAP` : quitter

```bash
python recognize.py
```
- `+` / `-` : ajuster le seuil (sensibilité)
- `Q` / `ÉCHAP` : quitter

Au premier lancement, macOS demande l'autorisation caméra
(**Réglages Système → Confidentialité et sécurité → Caméra**).

## Réglage

Le modèle donne une **distance** (affichée dans le carré) : plus elle est
basse, plus le visage ressemble au tien. On décide « c'est moi » si la
distance passe sous le **seuil**.

- seuil **bas** (ex. 50) = plus strict
- seuil **haut** (ex. 80) = plus permissif

Ajuste-le en direct avec `+` / `-`, ou change `THRESHOLD` dans `recognize.py`.

## Limites (à savoir)

- Reconnaissance **2D** (LBPH) : simple et rapide, mais moins précise que les
  modèles à embeddings, et vulnérable au *spoofing* (une photo peut tromper).
- Prochaine étape possible : **liveness** (détection de clignement) et/ou un
  modèle plus moderne (FaceNet/ArcFace) une fois les bases en place.
