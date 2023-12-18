#!/bin/bash

echo "TODO: fill in the docker run command"

echo "Démarrage de l'application Flask dans le conteneur Docker..."

docker run -d -p 8000:8000 --env COMET_API_KEY=$COMET_API_KEY ift6758/serving:1.0.0

