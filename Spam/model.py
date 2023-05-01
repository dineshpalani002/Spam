import pandas as pd
df = pd.read_csv('spam.csv',encoding='latin-1')


df.rename(columns={"v1":"class_label","v2":"message"},inplace=True)
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)

df.class_label.value_counts()
df["class_label"]=df["class_label"].apply(lambda x: 1 if x == "spam" else 0)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
                                            df['message'],
                                            df['class_label'], 
                                            test_size = 0.3, 
                                            random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(
                    lowercase=True,     
                    stop_words='english' 
                    )

x_train_transformed = vectorizer.fit_transform(x_train) 
x_test_transformed  = vectorizer.transform(x_test)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(x_train_transformed, y_train)

ytest_predicted_labels = classifier.predict(x_test_transformed)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from joblib import dump

# modell = "spam_model.pkl"
# dump(classifier,modell)
new_email="For your reminder tomorrow you have drive.  Refer mail which had sent on 30.03.2023. "
new_email_transformed=vectorizer.transform([new_email])
print(new_email)
new_email_transformed.toarray()

result=classifier.predict(new_email_transformed)
if result==1:
  print("Spam")
else:
  print("Ham")

print ('Accuracy Score :',accuracy_score(y_test, ytest_predicted_labels))
print(df)