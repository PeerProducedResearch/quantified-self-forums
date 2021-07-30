# Holistic stop words function
import re

import gensim
import spacy
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords


nlp = spacy.load('en_core_web_sm')

set1 = list(nlp.Defaults.stop_words)
set2 = list(STOPWORDS)
set3 = list(stopwords.words('english'))

extra_set = ['ill', 'hi', 'im', 'ive', 'dont', 'hello', 'hey', 'like', 'thanks', 'maybe', 'q', 'e g', 'think', 'good', 'com', 'hop',
             'thats', 'c', 'b', 'l', 'il', 'x', 'z', 'v', 'f', 'e', 'q', 'isnt', 'wont', 'yes', 'want', 'let', 'know', 'id', 'g', 'thing',
             'thank', 'come', 'mm', 'w', 'non', 'day', 'look', 'baby', 'n', 'lot', 'way', 'use', 'try', 'hour', 'couple', 'week', 'ago',
             'i', 'h', 'et', 'img', 'cool', 'year', 'u', 'need', 'nice', 'guy', 'boy', 'pm', '3rd', 'pretty', 'sure', 'bit', 'week', 'minute',
             'png', 'screen', 'shot', 'sans', 'serif', 'mon', 'mar', 'kb', 'gmt', 'arial', 'helvetica', 'feb', 'long', 'period', 'time', 'font',
             'size', 'medium', 'interested', 'dat', 'drank', 'dad', 'bottle', 'turn', 'number', 'everyday', 'follow', 'life', 'jar', 'j','www',
             'happen', 'yeah', 'wish', 'love', 'cheer', 'fun', 'later', 'bad', 'unfortunately', 'org', 'july', 'ki', 'february', 'month', 'saw',
             'potentially', '21st', 'hf', 'rr', 'lf', 'wait', 'yay', 'especially', 'feel', 'go', '100s', 'bin']

stop_words_list = set1 + set2 + set3 + extra_set


# remove stopwords
def remove_stopwords(text):
    slist = [word for word in text.split() if word not in stop_words_list]
    text = (" ").join(slist)
    return text


def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
        yield(sent)


dictionary = corpora.Dictionary(df.token_NN_text)

# filter words appear less than 15 docs & more than 0.5 documents & keep o
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# document to 'bow' BAG OF WORDS
corpus = [dictionary.doc2bow(doc) for doc in df.token_NN_text]

LDA = gensim.models.ldamodel.LdaModel

# Build LDA model

lda_model = LDA(corpus=corpus,
                id2word=dictionary,
                num_topics=30,
                random_state=50,
                update_every=1,
                chunksize=100,
                passes=10,
                # alpha = "auto"
                )
import pyLDAvis.gensim_models as gensimvis

vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis_data)

html_string = pyLDAvis.prepared_data_to_html(vis_data)
from streamlit import components

components.v1.html(html_string, width=1300, height=800, scrolling=True)


====


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
pyLDAvis.enable_notebook()

vis_data1 = gensimvis.prepare(ldamodel, doc_term_matrix, dictionary)
pyLDAvis.display(vis_data1)


html_string1 = pyLDAvis.prepared_data_to_html(vis_data1)
from streamlit import components
components.v1.html(html_string1, width=1300, height=800, scrolling=True)

============================================================================================

22222

# FUNCTIONS PREPROCESSING FOR TOPIC MODELLING

vectorizer = CountVectorizer(analyzer='word',
                             min_df=10,                        # minimum reqd occurences of a word
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

data_vectorized = vectorizer.fit_transform(df.no_sw_LDA_text)

# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=20,
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )

lda_output = lda_model.fit_transform(data_vectorized)

# Define Search Param
search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
# Init the Model
lda = LatentDirichletAllocation()
# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)
# Do the Grid Search
model.fit(data_vectorized)

# Best Model
best_lda_model = model.best_estimator_

pyLDAvis.enable_notebook()

plot = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
plot

html_string1 = pyLDAvis.prepared_data_to_html(plot)
from streamlit import components
components.v1.html(html_string1, width=1300, height=800, scrolling=True)