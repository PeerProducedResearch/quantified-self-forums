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

