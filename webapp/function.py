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


    def ner(text):
        doc = nlp(text)
        return [X.label_ for X in doc.ents]


    ent = qs_data['cleaned_text'].\apply(lambda x: ner(x))
    ent = [x for sub in ent for x in sub]

    # data vectorized
    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=10,  # minimum reqd occurences of a word
                                 stop_words='english',  # remove stop words
                                 lowercase=True,  # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 # max_features=50000,             # max number of uniq words
                                 )

    data_vectorized = vectorizer.fit_transform(qs_data.no_sw_LDA_text)

    # Build LDA Model
    lda_model = LatentDirichletAllocation(n_components=20,
                                          max_iter=10,  # Max learning iterations
                                          learning_method='online',
                                          random_state=100,  # Random state
                                          batch_size=128,  # n docs in each learning iter
                                          evaluate_every=-1,  # compute perplexity every n iters, default: Don't
                                          n_jobs=-1,  # Use all available CPUs
                                          )
    lda_output = lda_model.fit_transform(data_vectorized)

    # train model

    # Define Search Param
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
    # Init the Model
    lda = LatentDirichletAllocation()
    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)
    # Do the Grid Search
    model.fit(data_vectorized)
    # Defining Best Model
    best_lda_model = model.best_estimator_
    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(data_vectorized)

    plot = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
    plot

    html_string = pyLDAvis.prepared_data_to_html(plot)
    from streamlit import components

    components.v1.html(html_string, width=1300, height=800, scrolling=True)


    _______
    features = st.beta_container()

    with features:
        st.header('Features created')

        st.markdown('* **first feature:** NER')
        st.markdown('* **second feature:** Topic Modelling')
        st.markdown('* **Third feature:** Social Network Analysis')


        --------