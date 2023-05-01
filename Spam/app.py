from lib2to3.pgen2.pgen import DFAState
from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)

@app.route('/')


def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.rename(columns={"v1":"class_label","v2":"message"},inplace=True)
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
   
	# Features and Labels
	df["class_label"]=df["class_label"].apply(lambda x: 1 if x == "spam" else 0)

	
	
	
	X_train, X_test, y_train, y_test = train_test_split(df['message'],df['class_label'], test_size = 0.3, random_state = 0)
	
    # Extract Feature With CountVectorizer
	cv = CountVectorizer(lowercase=True,stop_words='english')
	x_train_transformed = cv.fit_transform(X_train)
	x_test_transformed  = cv.transform(X_test)# Fit the Data

	
	#Naive Bayes Classifier
	
	classifier = MultinomialNB()
	classifier.fit(x_train_transformed, y_train)
	
	classifier.predict(x_test_transformed)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = classifier.predict(vect)
	return render_template('home.html',prediction = my_prediction)

if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0',port =5000)