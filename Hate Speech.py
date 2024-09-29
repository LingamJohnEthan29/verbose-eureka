import inline as inline
import pandas as pd
import numpy as np
import regex as re
import nltk
nltk.download('stopwords')
import seaborn as sns
import matplotlib.pyplot as ply

import string
dataset = pd.read_csv("Hate_speech.csv")
dataset["labels"] = dataset["class"].map({0:"Hate Speech", 1:"Offensive", 2:"Neither hate nor offensive speech"})
data = dataset[["tweet","labels"]]

#To remove all the special characters and extracting only useful words
from nltk.corpus import stopwords #Stopwords include prepositions articles etc
stopwords = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")
def clean_data(text):
    text = str(text).lower()
    text = re.sub('https?://\S+|www.\.S+@','',text)
    text = re.sub('..\[.*?@\]','',text)
    text = re.sub('<_*?>+','',text)
    text = re.sub('[%s]' %re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    #Stopwords removed
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    #Stemming the text
    text = [stemmer.stem(word) for word in text.split(" ")]
    text = " ".join(text)
    return text
data['tweet'] = data['tweet'].apply(clean_data)
X = np.array(data['tweet'])
y = np.array(data['labels'])
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
cv = CountVectorizer()
X =  cv.fit_transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size=0.33,random_state=42)

#Building Machine Learning Model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)
Y_pred = dt.predict(X_test)
#We cant visualize it here
#So we can see the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
#print(cm)

#heatmap = sns.heatmap(cm,annot=True,fmt=" .1f",cmap="PiYG")
#print(heatmap)

sample = "I love Earth"
sample = clean_data(sample)
print(sample)
data1 = cv.transform([sample]).toarray()
print(dt.predict(data1))







