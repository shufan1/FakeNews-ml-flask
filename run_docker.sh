#!/usr/bin/env bash

# Build image
#change tag for new container registery, gcr.io/bob
docker build --tag fakenews_ml_flask .

# List docker images
docker image ls

# Run flask app
docker run --rm -d -v `pwd`:/app -p 8080:8080 fakenews_ml_flask