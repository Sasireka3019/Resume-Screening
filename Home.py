from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd
# print("Hello world")
import base64
import pickle
import streamlit as st
import os
import warnings
import fitz
st.set_page_config(page_title='Home', page_icon='', layout="wide", initial_sidebar_state="expanded", menu_items=None)
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

def extract_text(doc_file):
    """
    Extract text from the document, process it and return list of sentences in the document
    :parameter
    doc_file: URL of the document
    :return
    all_text: List of all the sentences in the document
    """
    print("Extracting text...")
    doc = fitz.open(doc_file)
    all_text = chr(12).join([page.get_text() for page in doc])
    all_text = all_text.split(".")
    l = len(all_text)
    x = 0
    while x < l:
        all_text[x] = (all_text[x].replace('\n', ''))+'.'
        if all_text[x].strip(' ') == ".":
            all_text.remove(all_text[x])
            l -= 1
        else:
            x += 1
    return all_text

def predict_output(input, name):
    categories = {6: 'Data Science', 12: 'HR', 0: 'Advocate', 1: 'Arts', 24: 'Web Designing', 16: 'Mechanical Engineer', 22: 'Sales', 14: 'Health and fitness', 5: 'Civil Engineer', 15: 'Java Developer', 4: 'Business Analyst', 21: 'SAP Developer', 2: 'Automation Testing', 11: 'Electrical Engineering', 18: 'Operations Manager', 20: 'Python Developer', 8: 'DevOps Engineer', 17: 'Network Security Engineer', 19: 'PMO', 7: 'Database', 13: 'Hadoop', 10: 'ETL Developer', 9: 'DotNet Developer', 3: 'Blockchain', 23: 'Testing'}
    transformed = word_vectorizer.transform([input])
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    pred = model.predict(transformed)
    st.text("Awesome your resume is much suitable for : ")
    res = ""
    for x in pred:
        st.text(categories[x]+" role")
        res = categories[x]
    if(name == 'Web Designing' and res == 'Arts'):
        st.success("Your resume is eligible")
    if((name == 'Python Developer' or name == 'Software Developer') and res == 'Data Science'):
        st.success("Your resume is eligible")
    elif(name == 'Software Developer' and res == 'Python Developer'):
        st.success("Your resume is eligible")
    elif(name == 'Software Developer' and res == 'Java Developer'):
        st.success("Your resume is eligible")
    elif(name == res):
        st.success("Your resume is eligible")
    else:
        st.error("Sorry, Your resume is not eligible for role you are applying for!")
def save_uploadedfile(uploadedfile):
    with open(os.path.join(os.getcwd() , uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return uploadedfile.name
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
name = ""
submit = None
datafile = "Sasireka_CV.pdf"
def st_ui():
    global submit
    global name
    global datafile
    original_title = '<p style="font-family:Georgia; color:purple; font-size: 45px;">Resume Screening</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    datafile = st.file_uploader(label='Your document will be processed', type=['png', 'jpg', 'pdf'],
                                accept_multiple_files=False, label_visibility="visible")
    if datafile is not None:
        file_details = {"FileName": datafile.name, "FileType": datafile.type}
        datafile = save_uploadedfile(datafile)
    else:
        datafile = "Sasireka_CV.pdf"
    with st.form("my_form"):
        options = ['Data Science','Software Developer','HR','Advocate','Arts','Web Designing','Mechanical Engineer','Sales','Health and fitness','Civil Engineer','Java Developer','Business Analyst','SAP Developer','Automation Testing','Electrical Engineering','Operations Manager','Python Developer','DevOps Engineer','Network Security Engineer','PMO','Database','Hadoop','ETL Developer','DotNet Developer','Blockchain','Testing']
        name = (st.selectbox('Enter the role : ',(options)))
        submit = st.form_submit_button(label="Submit")
st_ui()
doc_text = extract_text(datafile)
print(doc_text)
req = ''
for x in doc_text:
    req += x
if(submit):
    predict_output(req, name)