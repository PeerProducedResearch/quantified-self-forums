import streamlit as st
import spacy
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go

# containers
header = st.beta_container()
dataset = st.beta_container()

with header:
    st.title("NLP for Quantified-Self Forum")
    st.subheader("This project will give an overview of Forum's interactions and offer an interactive dashboard"
                 " to discover topics discussed about")

with dataset:
    st.header('QS forum dataset')
    st.text('The dataset was extracted from QS website using Python to parse the JSON files')

    qs_data = pd.read_csv('datasets/global_df')
    st.write(qs_data.head())
#

def main():
    """NLP APP"""


if st.checkbox("Documents Overview"):
    st.subheader("N of Docs created overtime")

    # NER

    #

    #

if __name__ == '__main__':
    main()
# NLP packages

def entity_analyzer(my_text):
    nlp = spacy.load('en')
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx]
    return entities


