import numpy as np
import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from sklearn.preprocessing import OneHotEncoder


class SpacyTextCategorizer:
    """

    Example
    -------
    sc = SpacyTextCategorizer('en_core_web_sm')
    sc.fit(X_train, y_train)
    sc.predict(X_test)
    """

    def __init__(self, language_model, batch_size=None, drop=0.2):
        self.nlp, self.text_cat = self._prepare(language_model)
        self.optimizer = None
        self.batch_size = batch_size or compounding(4., 32., 1.001)
        self.drop = drop
        self.encoder = OneHotEncoder(sparse=False, categories='auto', dtype=int)

    @property
    def classes_(self):
        if not hasattr(self.encoder, "categories_"):
            return list()
        return next(iter(self.encoder.categories_)).tolist()

    @staticmethod
    def _prepare(language_model):
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
        tmp_array = pd.Series(y).values.reshape(-1, 1)
        one_hot_encoded = self.encoder.fit_transform(tmp_array)
        labels = pd.DataFrame(one_hot_encoded,
                              columns=self.encoder.categories_[0])
        return list(labels.T.to_dict().values())

    def _inverse_transform(self, y):
        result = self.encoder.inverse_transform(y)
        return np.array(result).flatten().tolist()

    def fit(self, X, y):
        y_train = self._transform(y)
        labels = self.classes_
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

    def predict_proba(self, X_test):
        self.text_cat.model.use_params(self.optimizer.averages)
        cats_predicted = [doc.cats for doc in self.text_cat.pipe(X_test)]
        proba_df = pd.DataFrame(cats_predicted)
        proba_df.columns = self._inverse_transform(proba_df.columns)
        proba_df.columns = self.classes_
        return proba_df.to_numpy()

    def predict(self, X_test):
        tmp = self.predict_proba(X_test).T
        decision = np.where(tmp == tmp.max(), 1, 0).T
        return self._inverse_transform(decision)
