# general
import pandas as pd
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer, util
# import pickle
from streamlit_tags import st_tags


data = pd.read_csv('bgt_skills_5k.csv')
data = data['skills'].unique()
summary_tfidf = pd.read_csv("similarity_matrix.csv",index_col='Occupation')
index = faiss.read_index('bgt_top5k_skills')

st.title('Job Recommender')
st.subheader('Input a list of skills to get the top jobs recommended for you')


skill_input = st_tags(label='Enter List of Skills: ',text='enter to add more')
skill_chosen = []
list_of_skills = skill_input

model = SentenceTransformer('distilbert-base-nli-mean-tokens')


@st.cache
def search(query):
    query_vector = model.encode([query])
    k = 1
    top_k = index.search(query_vector, k)
    return [data[_id] for _id in top_k[1].tolist()[0]]


for skill in list_of_skills:
    query = skill
    skill_chosen.append(search(query)[0])


st.write('Skill Matches: ')
st.success(skill_chosen)


def ideal_job(df,skills_chosen):
    df_new = df[skills_chosen]
    df_new['count'] = df_new.count(axis=1)
    df_new['avg'] = df_new.mean(axis=1)

    df_new = df_new.sort_values(by=['count','avg'],ascending=False)
    df_new = df_new.head(30)

    bins = [0, 0.2, 0.6, 1]
    names = ['Low', 'Medium', 'High']

    for i in skills_chosen:
        df_new[i] = pd.cut(df_new[i], bins, labels=names)

    df_new = df_new[skills_chosen]

    return df_new


with st.spinner('Wait for it...'):
    time.sleep(2)

top_jobs = ideal_job(summary_tfidf,skill_chosen)

st.write('Top job recommendations: ')
st.dataframe(top_jobs.style.applymap(lambda x: "background-color: #74992e" if x=='High' else "background-color: rgba(255, 255, 128, .5)" if x=='Medium' else "background-color: hsla(50, 33%, 25%, .75)"))
