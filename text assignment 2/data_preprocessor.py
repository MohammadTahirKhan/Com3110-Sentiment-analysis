import pandas as pd
import string
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

class DataPreprocessor:
    """
    Data preprocessor for sentiment analysis
    """
    
    def __init__(self, num_classes):
        """
        Initialize data preprocessor

        Args:
            num_classes (int): number of classes
        """
        self.num_classes = num_classes
        self.CLASS_MAPPING = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
        self.current_id = None
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['\'s', '\'d', '\'ll', '\'re', '\'ve', '``', '\'\'', '...', '—', '’', '“', '”', '‘', '...', '--'])
        self.stop_words.update(set(string.punctuation.replace('!', '')))

    def preprocess(self, filename):
        """ 
        Load and preprocess data from file

        Args:
            filename (str): name of file to load data from

        Returns:
            ids (list): list of sentence ids
            data (list): list of preprocessed phrases
            labels (list): list of sentiment labels
        """
        # load data from file
        df = pd.read_csv(filename, sep='\t', header=0)

        ids = df['SentenceId'].tolist()
        data = []
        # preprocessing
        for phrase in df['Phrase'].tolist():
            processed_phrase = phrase.split(" ")
            processed_phrase = [word.lower() for word in processed_phrase if word.lower() not in self.stop_words]
            data.append(processed_phrase)

        labels = []
        if 'Sentiment' in df.columns:
            if self.num_classes == 5:
                labels = df['Sentiment'].tolist()
            else:
                labels = [self.CLASS_MAPPING[label] for label in df['Sentiment'].tolist()]

        return ids, data, labels

