from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Preprocessor:

    def tokenize(self):
        from nltk.tokenize import word_tokenize
        for i, tweet in tqdm(enumerate(self.data), 'Tokenization'):
            self.data[i] = word_tokenize(tweet.lower())
        return self.data

    # flitrage de mots utiles: optionelle
    def filter_inutiles(self):
        from nltk.corpus import stopwords
        import re
        stop = set(stopwords.words("english"))
        noise = ['user']
        for i, tweet in tqdm(enumerate(self.data), 'Filter useless words'):
            self.data[i] = [w for w in tweet if w not in stop and not re.match(r"[^a-zA-Z\d\s]+", w) and w not in noise]
        return self.data

        # je veux essayer le modèle avec la troncature, lemmatization ou aucune d'entre eux

    # choisir seulement UNE par fois
    def stem(self):
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        for i, tweet in tqdm(enumerate(self.data), 'Stemming'):
            for j, word in enumerate(tweet):
                self.data[i][j] = stemmer.stem(word)
        return self.data

    def lemmatize(self):
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        for i, tweet in tqdm(enumerate(self.data), 'Lemmatization'):
            for j, word in enumerate(tweet):
                self.data[i][j] = wnl.lemmatize(word, pos=self.get_pos(word))
        return self.data
