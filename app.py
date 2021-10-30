from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging

import mllib


app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

@app.route("/")
def home():
    html = f"<h3>Detect if a news is fake</h3>"
    return html.format(format)
    
@app.route("/predict/<title>")
def predict(title):
    """Predicts the Height of MLB Players"""
    clf,train_accur,test_accur=mllib.fit_model()
    print("training accuracy_score:",train_accur )
    print("test accuracy_score:", test_accur)

    # title= request.form['title']
    prediction = mllib.predict(clf,title)
    return jsonify({'prediction': prediction})
   
@app.route("/makepredict", methods=['POST'])
def makepredict():
    """Predicts the Height of MLB Players"""
    clf,train_accur,test_accur=mllib.fit_model()
    print("training accuracy_score:",train_accur )
    print("test accuracy_score:", test_accur)
    
    json_payload = request.json
    LOG.info(f"JSON payload: {json_payload}")
    title= request.form['title']
    prediction = mllib.predict(clf,title)
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
