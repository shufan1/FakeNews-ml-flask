from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging
from joblib import dump, load
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
    
@app.route("/trainmodel")
def trainmodel():
    clf,train_accur,test_accur=mllib.retrain_mllib()
    LOG.info("training accuracy_score: %f"%train_accur )
    LOG.info("test accuracy_score: %f"%test_accur)
    return {'message':"model trained",'training accuracy_score':train_accur,'test accuracy_score':test_accur }
   
@app.route("/makepredict", methods=['POST'])
def makepredict():
    clf = mllib.load_model(model="model.joblib")
    json_payload = request.json
    LOG.info(f"JSON payload: {json_payload}")
    title = json_payload['title']
    prediction = mllib.predict(clf,title)
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
