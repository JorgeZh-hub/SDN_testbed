#!/bin/bash

echo "Eliminando imágenes Docker <none>..."

# Lista y elimina las imágenes dangling
docker images -f "dangling=true" -q | xargs -r docker rmi

echo "Imágenes <none> eliminadas."
