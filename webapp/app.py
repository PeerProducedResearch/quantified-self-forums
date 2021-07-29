import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly_express as px

from gensim import corpora
import gensim
import pyLDAvis
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag, word_tokenize
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from spacy import displacy


# containers
header = st.beta_container()
dataset = st.beta_container()
interactive = st.beta_container()


#SIDE BARS
TM = st.sidebar.checkbox('Topic Modelling')
NER = st.sidebar.checkbox('NER: Named Entity Recognition')

# data sets functions
@st.cache
def get_data(filename):
    data = pd.read_csv(filename)
    return data


with header:
    st.title("NLP for Quantified-Self Forum")
    st.subheader("This project will give an overview of Forum's interactions and offer an interactive dashboard"
                 " to discover topics discussed about")

    st.image('Data_Viz/qs_wordcloud.png')
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

# Example for detecting bigrams
bigram_measures = nltk.collocations.BigramAssocMeasures()

finder =nltk.collocations.BigramCollocationFinder\
.from_documents([comment.split() for comment in\
                 df.no_sw_LDA_text])


# Filter only those that occur at least 50 times
finder.apply_freq_filter(50)
bigram_scores = finder.score_ngrams(bigram_measures.pmi)


trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = nltk.collocations.TrigramCollocationFinder.from_documents([comment.split() for comment in df.no_sw_LDA_text])
# Filter only those that occur at least 50 times
finder.apply_freq_filter(50)
trigram_scores = finder.score_ngrams(trigram_measures.pmi)

bigram_pmi = pd.DataFrame(bigram_scores)
bigram_pmi.columns = ['bigram', 'pmi']
bigram_pmi.sort_values(by='pmi', axis=0, ascending=False, inplace=True)

trigram_pmi = pd.DataFrame(trigram_scores)
trigram_pmi.columns = ['trigram', 'pmi']
trigram_pmi.sort_values(by='pmi', axis=0, ascending=False, inplace = True)

# Filter for bigrams with only noun-type structures
@st.cache
def bigram_filter(bigram):
    tag = nltk.pos_tag(bigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['NN']:
        return False
    if 'n' in bigram or 't' in bigram:
        return False
    if 'PRON' in bigram:
        return False
    return True

             # Filter for trigrams with only noun-type structures
@st.cache
def trigram_filter(trigram):
    tag = nltk.pos_tag(trigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['JJ','NN']:
        return False
    if 'n' in trigram or 't' in trigram:
         return False
    if 'PRON' in trigram:
        return False
    return True

 # choose top 500 ngrams in this case ranked by PMI that have noun like structures
filtered_bigram = bigram_pmi[bigram_pmi.apply(lambda bigram:\
                                              bigram_filter(bigram['bigram'])\
                                              and bigram.pmi > 5, axis = 1)][:500]

filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram: \
                                                 trigram_filter(trigram['trigram'])\
                                                 and trigram.pmi > 5, axis = 1)][:500]


bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2]
trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]

      # Concatenate n-grams
@st.cache
def replace_ngram(x):
    for gram in trigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    for gram in bigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    return x

reviews_w_ngrams = df.copy()
reviews_w_ngrams.no_sw_LDA_text = reviews_w_ngrams.no_sw_LDA_text.map(lambda x: replace_ngram(x))

# tokenize reviews + remove stop words + remove names + remove words with less than 2 characters
reviews_w_ngrams = reviews_w_ngrams.no_sw_LDA_text.map(lambda x: [word for word in x.split()\
                                                 if  len(word) > 2])

# Filter for only nouns
@st.cache
def noun_only(x):
    pos_comment = nltk.pos_tag(x)
    filtered = [word[0] for word in pos_comment if word[1] in ['NN']]
    return filtered

final_posts = reviews_w_ngrams.map(noun_only)

# TOPIC MODELLING
dictionary = corpora.Dictionary(final_posts)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in final_posts]

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=40,\
               iterations=200,  chunksize = 10000, eval_every = None, random_state=0)

import pyLDAvis.gensim_models as gensimvis

vis_data1 = gensimvis.prepare(ldamodel, doc_term_matrix, dictionary)
pyLDAvis.display(vis_data1)


html_string1 = pyLDAvis.prepared_data_to_html(vis_data1)
from streamlit import components                                         
components.v1.html(html_string1, width=1300, height=800, scrolling=True)