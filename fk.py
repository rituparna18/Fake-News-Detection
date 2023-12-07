# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
import re
import string
import os

# %%
data = pd.read_csv("/FakeNews/data.csv")
data.head(10)

# %%
def wordopt (text):
    text= text.lower()
    text= re.sub('\[.*?\]','',text)
    text= re.sub("\\W"," ", text)
    text = re.sub ('https?://\S+|www\.\S+', '', text)
    text= re.sub ('<.*?>+','', text)
    text= re.sub ('[%s]' % re.escape(string.punctuation),'',text)
    text= re.sub('\n', '', text)
    text= re.sub('\w*\d\w*','',text)
    return text

def getConfusionMatrix(y_test,y_pred):
    conf_mat = confusion_matrix(y_test,y_pred)

    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    return plt

# %%
data['text'] = data['text'].apply(wordopt)
x=data['text']
y=data['class']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# %%
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)
print(LR.score(xv_test, y_test))


# %%
plt = getConfusionMatrix(y_test,pred_lr)
plt.show()
print(classification_report(y_test, pred_lr))

# %%
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)
print(DT.score(xv_test, y_test))

# %%
plt = getConfusionMatrix(y_test,pred_dt)
plt.show()
print(classification_report(y_test, pred_dt))

# %%
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)

pred_gb = GB.predict(xv_test)
print(GB.score(xv_test, y_test))

# %%
plt = getConfusionMatrix(y_test,pred_gb)
plt.show()
print(classification_report(y_test,pred_gb))

# %%
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)

pred_rf = RF.predict(xv_test)
print(GB.score(xv_test,y_test))


# %%
plt = getConfusionMatrix(y_test,pred_rf)
plt.show()
print(classification_report(y_test,pred_rf))


# %%
def output_lable(n):
  if n == 0:
    return "Fake News"
  elif n == 1:
    return "Not A Fake News"
def manual_testing(news):
  testing_news = {"text": [news]}
  new_def_test = pd.DataFrame (testing_news)
  new_def_test["text"] = new_def_test["text"].apply(wordopt)
  new_x_test = new_def_test["text"]
  new_xv_test=vectorization. transform(new_x_test)
  pred_LR = LR.predict(new_xv_test)
  pred_DT = DT.predict(new_xv_test)
  pred_GB = GB.predict(new_xv_test)
  # pred_RF = RF.predict(new_xv_test)
  return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {}".format(output_lable(pred_LR[0]),output_lable(pred_DT[0]),output_lable(pred_GB[0])))

# %%
news=str(input())
manual_testing(news)


