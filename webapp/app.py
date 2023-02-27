import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly_express as px

from gensim import corpora
import gensim
import pyLDAvis
import nltk
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag, word_tokenize
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from spacy import displacy
from PIL import Image


# containers
header = st.container()
dataset = st.container()
interactive = st.container()


#SIDE BARS
TM = st.sidebar.checkbox('Topic Modelling')
NER = st.sidebar.checkbox('NER: Named Entity Recognition')

# data sets functions
@st.cache_data
def get_data(filename):
    data = pd.read_csv(filename)
    return data


with header:
    st.title("NLP for Quantified-Self Forum")
    st.subheader("Self-Quantified is a Community website of self-trackers, self-researchers interested in personal" 
                 " science. It encourages all types of researches to post about their research, ask for advices"
               "  and explore different methods of self-tracking through wearables."
    
                 " This project will analyze  the Website Forum's interactions and offer an interactive dashboard"
                 " to discover topics discussed about")

  #  image = Image.open('./Data_Viz/qs_wordcloud.png')

   # st.image(image, caption='Wordcloud of Quantified self forum topics')
    st.sidebar.title("Side Menu")
    st.sidebar.write("-----------")
    st.sidebar.write("**Visualizations:** ")


with dataset:
    st.header('QS forum dataset')
    st.text('The dataset was extracted from QS website using Python to parse the JSON files')

    qs_data = get_data('https://media.githubusercontent.com/media/KaoutarLanjri/large_files/master/global_df.csv')
    st.write(qs_data.head())

    st.subheader('Distribution of Yearly Created Posts')
    docs_list = pd.DataFrame(qs_data['date_year'].value_counts())
    st.bar_chart(docs_list)

with interactive:
    st.title('Closer look into the data')

    fig = go.Figure(data=go.Table(
                columnwidth=[1, 1, 3, 3],
                header=dict(values=list(qs_data[['topic_id', 'creation_date', 'noHTML_text', 'lemmat_text']].columns),
                fill_color='#FD8E72',
                align='center'),
                cells=dict(values=[qs_data.topic_id, qs_data.creation_date, qs_data.noHTML_text, qs_data.lemmat_text])))

    fig.update_layout(margin=dict(l=5, r=5, b=10, t=10))
    st.write(fig)

    # LINE CHART for Word occurence over time*

# ALL TIME
    words_freq = get_data(
        'https://raw.githubusercontent.com/KaoutarLanjri/quantified-self-forums/main/datasets/words_df.csv')

    fig = px.line(words_freq, x=words_freq.creation_year, y=words_freq.columns[0:30],
                  title="Word Dispersion over time 2011 to 2021")

    fig.update_xaxes(type='category')
    st.write(fig)

# 2021
    words21_freq = get_data(
        'https://raw.githubusercontent.com/KaoutarLanjri/quantified-self-forums/main/datasets/words_df_2021.csv')

    fig = px.line(words21_freq, x=words21_freq.creation_date, y=words21_freq.columns[0:30],
                  title="Word Dispersion over the year 2021")

    fig.update_xaxes(type='category')
    st.write(fig)

    st.header('Topic Modelling')
    st.image('Data_Viz/coherence_score_chart.png')
    st.text('The improvement stops significantly improving after 9 topics')

# TOPIC MODELLING
    st.header('While Topic Modelling plot is loading...'
              'Check the topics keywords table')

    st.image('Data_Viz/topic_model_words.png')
    st.text('plot loading...')
# Tokenize Text
def tokenize_text(text):
    filtered_text = []
    words = word_tokenize(text)

    for word in words:
        filtered_text.append(word)
    return filtered_text
#
df = qs_data.copy()

df['no_sw_LDA_text'] = df['no_sw_LDA_text'].astype('str')

df = df.groupby(['topic_id'], as_index=False).agg({('no_sw_LDA_text'): ' '.join})

df['token_NN_text'] = df.no_sw_LDA_text.apply(lambda x: tokenize_text(x))


# FUNCTIONS PREPROCESSING FOR TOPIC MODELLING
dictionary = corpora.Dictionary(df.token_NN_text)

# filter words appear less than 15 docs & more than 0.5 documents & keep only 100k most frequent words
dictionary.filter_extremes(no_below=20, no_above=0.5)
# document to 'bow' BAG OF WORDS
corpus = [dictionary.doc2bow(doc) for doc in df.token_NN_text]

# MODEL
Lda = gensim.models.ldamodel.LdaModel

ldamodel = Lda(corpus, num_topics=11, id2word = dictionary, passes=10,\
               iterations=100,  chunksize =1000, random_state=100, eval_every = None, alpha='auto', eta='auto')

from pyLDAvis import gensim
from gensim.models import LdaModel, HdpModel

vis_data1 = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis_data1)

html_string1 = pyLDAvis.prepared_data_to_html(vis_data1)
from streamlit import components
components.v1.html(html_string1, width=1300, height=800, scrolling=True)
