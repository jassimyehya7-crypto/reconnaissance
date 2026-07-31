#!/bin/bash
# Etape 3 : lancer la surveillance (identite + posture + alerte).
# (Identique au fichier 'reconnaissance facial' de la racine.)
cd "$(dirname "$0")/.." || exit 1

if [ ! -d venv ]; then
  echo "Installe d'abord en double-cliquant 'reconnaissance facial' a la racine."
  read -r -p "Entree pour quitter..."
  exit 1
fi

source venv/bin/activate
echo "Surveillance... Q pour quitter."
python watch.py
