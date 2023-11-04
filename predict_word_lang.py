def predict_word_lang(word: str | bool, return_label: str|bool=True, model: str="logreg"):
    """takes word as input, 
    True|False|'both' are passed to return_label to respectively return: language label, probability list, sorted mapping of labels and probabilities as output,
    model parameter takes aliases of pre-trained models ('logreg' or 'bayes') or a url-link to the model serialized in a pickle format, then opens the model and use it for predictions
    """
    import joblib
    from urllib.request import urlopen
    # aliases to download specific models
    if model == "logreg": 
        model = joblib.load(urlopen("https://github.com/LingConLab/Cyrillic_languages_identification_models_and_data_stroe/raw/main/models/logreg_model.pickle"))
    elif model == "bayes":
        model = joblib.load(urlopen("https://github.com/LingConLab/Cyrillic_languages_identification_models_and_data_stroe/raw/main/models/bayes_model.pickle"))
    else: 
        model = joblib.load(urlopen(model)) # any other link to a *raw* file in input
    
    """sklearn models can predict only on array objects, that's why we cover a single-word
    input string into a list (square brackes)"""

    if return_label is True:
        return model.predict([str(word)])
    if return_label is False:
        model.predict_proba([str(word)])[0] #index 0 to return python list, remove for np.array

    if return_label == "both":
        # return a dictionary with language-probability correspondences
        # sorted with descending probability
        proba_map = dict(map(lambda i,j: (i,j),
                             model.classes_,# keys 
                             model.predict_proba([str(word)])[0], # values
                             ))
        
        return dict(sorted(proba_map.items(), key=lambda x: x[1], reverse=True))