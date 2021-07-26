import spacy

def entity_analyzer(my_text):
    nlp = spacy.load('en')
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx]
    return entities


with interactive:
    st.title('Closer look into the data')

    fig = go.Figure(data=go.Table())
    fig.update_layout()
    st.write(fig)