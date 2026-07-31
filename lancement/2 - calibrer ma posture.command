#!/bin/bash
# Etape 2 : calibrer debout / assis. Double-clique ce fichier.
cd "$(dirname "$0")/.." || exit 1

if [ ! -d venv ]; then
  echo "Installe d'abord en double-cliquant 'reconnaissance facial' a la racine."
  read -r -p "Entree pour quitter..."
  exit 1
fi

source venv/bin/activate
echo "Calibrage : D=debout, A=assis, S=sauver, Q=quitter."
python calibrate_posture.py
