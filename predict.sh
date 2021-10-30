#!/usr/bin/env bash

PORT=8080
echo "Port: $PORT"

# POST method predict
curl -d '{  
   "title":" Donald Trump Sends Out Embarrassing New Year?€?s Eve Message; This is Disturbing"
}'\
     -H "Content-Type: application/json" \
     -X POST http://localhost:$PORT/predict