import streamlit as st
import spacy
import pandas as pd
import plotly.graph_objects as go

# containers
header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
interactive = st.beta_container()

# to make webapp quicker
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

    qs_data = get_data('https://media.githubusercontent.com/media/KaoutarLanjri/large_files/master/global_df.csv')
    st.write(qs_data.head())

    st.subheader('Distribution of Yearly Created Posts')
    docs_list = pd.DataFrame(qs_data['date_year'].value_counts())
    st.bar_chart(docs_list)

with features:
    st.header('Features created')

    st.markdown('* **first feature:** NER')
    st.markdown('* **second feature:** Topic Modelling')
    st.markdown('* **Third feature:** Social Network Analysis')

with interactive:
    st.title('Closer look into the data')

    fig = go.Figure(data=go.Table
        (header=dict(values=list(qs_data[['topic_id', 'creation_date', 'lemmat_text']].columns),
        fill_color='#FD8E72',
        align='center'),
    cells=dict(values=[qs_data.topic_id, qs_data.creation_date, qs_data.lemmat_text]
               )))
    #fig.update_layout()
    st.write(fig)

def main():
    """NLP APP"""


if st.checkbox("Documents Overview"):
    st.subheader("N of Docs created overtime")

    # NER

    #

    #

if __name__ == '__main__':
    main()