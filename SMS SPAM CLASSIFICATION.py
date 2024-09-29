import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#####IMPORTING DATASET
encodings = ['utf-8', 'ISO-8859-1', 'latin1']
for encoding in encodings:
    try:
        dataset = pd.read_csv("spam.csv", encoding=encoding,sep=',')
        #print("File read successfully with encoding:", encoding)
        break
    except UnicodeDecodeError:
        print("Error decoding file with encoding:", encoding)
dataset['v1'] = dataset['v1'].map({'ham':0,'spam':1})
###Using visualization from matplotlib we can see that the dataset is imbalanced
##Handling imbalanced dataset using Oversampling
only_spam = dataset[dataset['v1']==1]
count= int((dataset.shape[0] - only_spam.shape[0]) / only_spam.shape[0])
for i in range(0,count-1):
    dataset = pd.concat([dataset,only_spam]) ##DOES CONCAT 6 TIMES TO BRING A MORE BALANCED DATASET

###CREATING A NEW FEATURE WORD_COUNT
dataset['word_count'] = dataset['v2'].apply(lambda x: len(x.split()))

###CREATING FEATURE FOR PRESENCE OF $ SIGN
def currency_present(data):
    currency_symbols = ['$',]
    for i in currency_symbols:
        if i in dataset:
            return 1
    return 0

dataset['currency_symbols contained'] = dataset['v2'].apply(currency_present)
##FEATURE TO DETECT NUMBERS
def number_detect(data):
    for i in data:
        if (ord(i) >= 40 and ord(i) <= 57):
            return 1
    return 0
dataset['contains_number'] = dataset['v2'].apply(number_detect)


####DATA CLEANING
import nltk
import regex as re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
corpus = []
wnl = WordNetLemmatizer()

for sms in list(dataset.v2):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms) #FILTERING NUMBERS AND SPECIAL CHARACTERS
    message = message.lower()
    words = message.split()
    #STOPWORD REMOVAL
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]
    message = ' '.join(lemm_words)
    corpus.append(message)

##Creating Bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 500)
vectors = tfidf.fit_transform(corpus).toarray()
features_names =  tfidf.get_feature_names_out()
x = pd.DataFrame(vectors, columns = features_names)
y = dataset['v1']
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

###NAIVE BAYES MODEL
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
cv = cross_val_score(mnb, x, y, scoring='f1', cv=10)
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
#print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test, y_pred)
#print("The confusion matrix of Naive Bayes is", cm)

####DECISION TREE MODEL
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
cv1 = cross_val_score(dt, x, y, scoring='f1',cv=10)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
#print(classification_report(y_test,y_pred))
cm1 = confusion_matrix(y_test, y_pred)
#print("The confusion matrix of DecisionTreeClassifier is", cm1)
def predict_spam(SMS):
        message = re.sub(pattern='[^a-zA-Z]',repl = ' ',string=SMS) #FILTERING NUMBERS AND SPECIAL CHARACTERS
        message = message.lower()
        words = message.split()
    #STOPWORD REMOVAL
        filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
        lemm_words = [wnl.lemmatize(word) for word in filtered_words]
        message = ' '.join(lemm_words)
        predict_this = tfidf.transform([message]).toarray()
        #print("From DecisionTree",dt.predict(predict_this))
        if mnb.predict(predict_this) == 0:
            print("Not spam")
        else:
            print("Spam")

msg = input("Spam message")
predict_spam(msg)


