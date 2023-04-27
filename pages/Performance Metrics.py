import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle
import base64
st.set_page_config(page_title='Performance Metrics', page_icon='', layout="wide", initial_sidebar_state="expanded", menu_items=None)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg.jpg')
resumeDataSet = pd.read_csv('cleanedResume.csv' ,encoding='utf-8')
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english')
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

# print ("Feature completed .....")

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2, shuffle=True, stratify=requiredTarget)

filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

original_title = '<p style="font-family:Georgia; color:black; font-size: 25px;">Model Accuracy</p>'
st.markdown(original_title, unsafe_allow_html=True)
prediction = model.predict(X_test)
score = accuracy_score(y_test, prediction)
st.success(score)
original_title = '<p style="font-family:Georgia; color:black; font-size: 25px;">Classification Report</p>'
st.markdown(original_title, unsafe_allow_html=True)
#st.text(metrics.classification_report(y_test, prediction))

metrics = metrics.classification_report(y_test, prediction)
lst = metrics.split("\n")
headers = []
for x in range(1):
    f = lst[x].split(" ")
    for y in f:
        if(y != ''):
            headers.append(y)
data = []
for x in range(2, 27):
    f = lst[x].split()
    temp = []
    for y in f:
        if(y != ''):
            temp.append(y)
    data.append(temp[1:])
df = pd.DataFrame(data, columns=headers)
st.table(df)