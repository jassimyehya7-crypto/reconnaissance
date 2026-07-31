#!/bin/bash
# ============================================================
#  RECONNAISSANCE FACIALE
#  Double-clique ce fichier pour lancer la surveillance.
# ============================================================

# Se placer dans le dossier du projet (là où est ce fichier)
cd "$(dirname "$0")" || exit 1

echo "== Reconnaissance faciale =="

# Premiere fois : installer l'environnement automatiquement
if [ ! -d venv ]; then
  echo "Premiere installation (une seule fois), patiente..."
  python3 -m venv venv || { echo "Python 3 introuvable."; read -r -p "Entree pour quitter..."; exit 1; }
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

# Verifier que le visage et la posture sont enregistres
if [ ! -f known_face.npz ]; then
  echo ""
  echo "!! Ton visage n'est pas encore enregistre."
  echo "   Ouvre le dossier 'lancement' et double-clique :"
  echo "   '1 - enregistrer mon visage'"
  read -r -p "Entree pour quitter..."
  exit 0
fi
if [ ! -f posture.npz ]; then
  echo ""
  echo "!! Ta posture n'est pas encore calibree."
  echo "   Ouvre le dossier 'lancement' et double-clique :"
  echo "   '2 - calibrer ma posture'"
  read -r -p "Entree pour quitter..."
  exit 0
fi

echo "Lancement... (appuie sur Q dans la fenetre pour quitter)"
python watch.py
