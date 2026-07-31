# reconnaissance

Prototype de reconnaissance faciale sur webcam (Mac), en **OpenCV pur**.
Un carré « scanne » ta tête en direct : **vert** si ça te reconnaît, **rouge** sinon.

Détection **YuNet** + reconnaissance **SFace** (empreinte 128-D, similarité cosinus).
Pas de Homebrew, pas de cmake, pas de compilation.

## Aperçu

- `face_utils.py` — outils partagés + téléchargement auto des modèles.
- `enroll.py` — étape 1 : tu enregistres ton visage une fois (crée `known_face.npz`).
- `recognize.py` — étape 2 : reconnaissance en direct avec le carré vert/rouge.

## Installation (macOS)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Au premier lancement, les 2 modèles (~39 Mo) se téléchargent tout seuls dans `models/`.

## Utilisation

```bash
python enroll.py
```
- `ESPACE` : démarrer/arrêter la capture automatique (vise ~20 empreintes)
- `C` : capturer une seule empreinte
- `S` / `ENTRÉE` : sauvegarder
- `Q` / `ÉCHAP` : quitter

```bash
python recognize.py
```
- `+` / `-` : ajuster le seuil (sensibilité)
- `Q` / `ÉCHAP` : quitter

macOS demande l'autorisation caméra au premier lancement
(**Réglages Système → Confidentialité et sécurité → Caméra**).

## Réglage

Le score affiché est une **similarité cosinus** : plus il est **haut**, plus
le visage te ressemble. On décide « c'est moi » si le score dépasse le **seuil**
(par défaut `0.36`, la valeur recommandée par OpenCV).

- seuil **haut** (ex. 0.45) = plus strict
- seuil **bas** (ex. 0.30) = plus permissif

Ajuste-le en direct avec `+` / `-`.

## Limites (à savoir)

- Reconnaissance **2D** : bien plus précise que LBPH, mais toujours vulnérable
  au *spoofing* (une photo peut tromper). Pas d'équivalent Face ID sans caméra
  infrarouge / capteur de profondeur.
- Prochaine étape possible : **liveness** (détection de clignement) pour qu'une
  simple photo ne passe pas au vert.
