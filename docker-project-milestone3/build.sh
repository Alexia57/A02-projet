#!/bin/bash

# echo "TODO: fill in the docker build command"

echo "Construction de l'image Docker pour l'application Flask en cours..."

docker build -t ift6758/serving:1.0.0 .

# Afficher un message de succès
echo "L'image Docker 'flask-app' a été construite avec succès."#!/bin/bash

