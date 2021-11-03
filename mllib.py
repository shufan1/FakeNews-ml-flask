import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from joblib import load, dump

nltk.download('wordnet')
nltk.download('stopwords')
ps = WordNetLemmatizer()
stopwords = stopwords.words('english')
vectorizer = TfidfVectorizer(max_features = 50000 , lowercase=False , ngram_range=(1,2))

def load_data():
    true_data = pd.read_csv('True.csv')
    fake_data = pd.read_csv('Fake.csv')
    true_data.head()
    true_data['label']=[1]*len(true_data)
    fake_data['label']=[0]*len(fake_data)
    data=true_data.append(fake_data).sample(frac=1).reset_index().drop(columns=['index'])
                                                                    
    return data
    
def token_clean(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]',' ',row)
    token= row.split()
    news = [ps.lemmatize(word) for word in token if not word in stopwords]
    cleaned_news=' '.join(news)
    return cleaned_news
    

def prep_data(data):
    data = data[:3000]
    train_text, test_text, train_labels, test_labels = train_test_split(data['title'], data['label'], 
                                                                random_state=2018, 
                                                                test_size=0.3)
    train_text_cleaned = list(map(lambda row: token_clean(row), train_text))
    test_text_cleaned = list(map(lambda row: token_clean(row), test_text))
    
    
    return train_text_cleaned,test_text_cleaned,train_labels,test_labels
    
    
def vectorize(X_train,X_test):
    # print(X_train)
    # use tfidf approch to vecotrize the text 
    vec_train_data = vectorizer.fit_transform(X_train) 
    vec_train_data = vec_train_data.toarray()
    vec_test_data = vectorizer.transform(X_test).toarray()
    
    training_data = pd.DataFrame(vec_train_data , columns=vectorizer.get_feature_names_out())
    testing_data = pd.DataFrame(vec_test_data , columns= vectorizer.get_feature_names_out())
    
    return training_data,testing_data


def fit_model():
    data = load_data()
    X_train,X_test,y_train,y_test = prep_data(data)
    X_train_vect,X_test_vect = vectorize(X_train,X_test)
    
    clf = MultinomialNB()
    clf.fit(X_train_vect, y_train)
    y_pred_train = clf.predict(X_train_vect)
    y_pred  = clf.predict(X_test_vect)
    train_accur= accuracy_score(y_train , y_pred_train)
    test_accur =accuracy_score(y_test, y_pred)
    # log train accur and test_accur somewhere
    return clf,train_accur,test_accur

def retrain_mllib():
    #retrian and export the library
    clf,train_accur,test_accur = fit_model()
    dump(clf, 'model.joblib')
    return clf,train_accur,test_accur
    
def load_model(model="model.joblib"):
    """Grabs model from disk"""
    clf = load(model)
    return clf


def predict(clf,news):
    news_cleaned = token_clean(news)
    news_vect = vectorizer.transform([news_cleaned]).toarray()
    case = pd.DataFrame(news_vect, columns=vectorizer.get_feature_names_out())
    single_prediction = clf.predict(case)
    predict_data = {
        "News title": news_cleaned,
        "True news": "True" if single_prediction==1 else "Fake",
    }

    # logging.debug(f"Prediction: {predict_log_data}")
    return predict_data
    

    
    
if __name__=="__main__":
    clf,train_accur,test_accur=fit_model()
    print("training accuracy_score:",train_accur )
    print("test accuracy_score:", test_accur)
    single_prediction = predict(clf,"U.S. military to accept transgender recruits on Monday: Pentagon")#"Imposters posing as army personnel on the social media have been called out by the Indian Army as false news and disinformation.")
    print(single_prediction)

