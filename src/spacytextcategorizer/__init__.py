import numpy as np
import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from sklearn.preprocessing import OneHotEncoder


class SpacyTextCategorizer:
    """

    Example
    -------
    sc = SpacyCategorizer('en_core_web_sm')
    sc.fit(X_train, y_train)
    sc.predict(X_test)
    """

    def __init__(self, language_model, batch_size=None, drop=0.2, tokenizer=None):
        self.nlp, self.text_cat = self._prepare(language_model)
        self.optimizer = None
        self.batch_size = batch_size or compounding(4., 32., 1.001)
        self.drop = drop
        self.tokenizer = tokenizer or self.nlp.tokenizer
        # optional
        self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto', dtype=int)

    def _prepare(self, language_model):
        """Prepare the language model and text categorizer.

        :param language_model:
        :type language_model:
        :param pip_name:
        :type pip_name:
        :param labels:
        :type labels:
        :return:
        :rtype:
        """
        pip_name = 'textcat'
        nlp = spacy.load(language_model)
        # add the text classifier to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if pip_name not in nlp.pipe_names:
            text_cat = nlp.create_pipe(pip_name)
            nlp.add_pipe(text_cat, last=True)
        # otherwise, get it, so we can add labels to it
        else:
            text_cat = nlp.get_pipe(pip_name)

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
        nlp.disable_pipes(*other_pipes)
        return nlp, text_cat

    def _transform(self, y):
        temp_df = y.values.reshape(-1, 1)
        onehot_encoded = self.onehot_encoder.fit_transform(temp_df)
        labels = pd.DataFrame(onehot_encoded,
                              columns=self.onehot_encoder.categories_[0])
        return list(labels.T.to_dict().values())

    def _inverse_transform(self, y):
        return self.onehot_encoder.inverse_transform(y)

    def fit(self, X, y):
        y_train = self._transform(y)
        labels = self.onehot_encoder.categories_[0]
        for category in labels:
            self.text_cat.add_label(category)
        train_data = list(zip(X, [{'cats': cats} for cats in y_train]))
        self.optimizer = self.nlp.begin_training()
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(train_data, size=self.batch_size)
        for batch in batches:
            texts, annotations = zip(*batch)
            self.nlp.update(texts, annotations, sgd=self.optimizer, drop=self.drop,
                            losses=losses)
        return self

    def predict(self, X_test):
        self.text_cat.model.use_params(self.optimizer.averages)
        docs = (self.tokenizer(text) for text in X_test)
        cats_predicted = [doc.cats for doc in self.text_cat.pipe(docs)]
        tmp = pd.DataFrame(cats_predicted, columns=cats_predicted[0].keys())
        new_df = pd.DataFrame(np.where(tmp.T == tmp.T.max(), 1, 0), index=tmp.columns).T
        return self._inverse_transform(new_df)
