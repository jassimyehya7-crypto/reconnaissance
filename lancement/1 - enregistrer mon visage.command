#!/bin/bash
# Etape 1 : enregistrer ton visage. Double-clique ce fichier.
cd "$(dirname "$0")/.." || exit 1

if [ ! -d venv ]; then
  echo "Installe d'abord en double-cliquant 'reconnaissance facial' a la racine."
  read -r -p "Entree pour quitter..."
  exit 1
fi

source venv/bin/activate
echo "Enrolement du visage : ESPACE pour capturer, S pour sauver, Q pour quitter."
python enroll.py
