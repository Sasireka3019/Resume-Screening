import streamlit as st
import matplotlib.pyplot as plt
st.set_page_config(page_title='Data Visualization', page_icon='', layout="wide", initial_sidebar_state="expanded", menu_items=None)
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.gridspec import GridSpec

resumeDataSet = pd.read_csv('Resume.csv' ,encoding='utf-8')
fig = plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
categories = ['Data Science','Software Developer','HR','Advocate','Arts','Web Designing','Mechanical Engineer','Sales','Health and fitness','Civil Engineer','Java Developer','Business Analyst','SAP Developer','Automation Testing','Electrical Engineering','Operations Manager','Python Developer','DevOps Engineer','Network Security Engineer','PMO','Database','Hadoop','ETL Developer','DotNet Developer','Blockchain','Testing']
ax=sns.countplot(x="Category", data=resumeDataSet)
# plt.xlabel(categories)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.grid()
st.pyplot(fig)
