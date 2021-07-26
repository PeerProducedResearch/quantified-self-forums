import streamlit as st
import spacy
import pandas as pd
import plotly.graph_objects as go

# containers
header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()

@st.cache
def get_data(filename):
    qs_data = pd.read_csv(filename)
    return qs_data

with header:
    st.title("NLP for Quantified-Self Forum")
    st.subheader("This project will give an overview of Forum's interactions and offer an interactive dashboard"
                 " to discover topics discussed about")

with dataset:
    st.header('QS forum dataset')
    st.text('The dataset was extracted from QS website using Python to parse the JSON files')

    qs_data = get_data('/home/criuser/nlp/a_Quantified_Self_Project/global_df.csv')
    st.write(qs_data.head())

    st.subheader('Distribution of Yearly Created Posts')
    docs_list = pd.DataFrame(qs_data['date_year'].value_counts())
    st.bar_chart(docs_list)

with features:
    st.header('Features created')

    st.markdown('* **first feature:** NER')
    st.markdown('* **second feature:** Topic Modelling')
    st.markdown('* **Third feature:** Social Network Analysis')


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


