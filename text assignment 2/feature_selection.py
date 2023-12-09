# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
import string

class FeatureSelection:
    def __init__(self, data):
        # self.sentence = sentence
        # self.stopwords = set(stopwords.words('english'))
        # self.PUNCTUATIONS = ['.', ',', '?', '!', ':', ';', '-lrb-', '-rrb-', '-lsb-', '-rsb-', '-lcb-', '-rcb-', '-', '--', '_', '`', '``', '\'', '\'\'', '...', '—']
        self.features = ['stopwords', 'punctuations', 'negation', 'binarization', 'adjectives', 'adverbs', 'nouns', 'verbs', 'bigrams', 'trigrams']
        # self.feature_selection = None
        self.processed_data_with_features = self.process_data_with_features(self, data)
        self.data = data
        
    def process_data_with_features(self, data):
        processed_data_with_features = []
        for sentence in self.data:
            self.preprocess_sentence_with_features(sentence)
            processed_data_with_features.append(sentence)
        return processed_data_with_features
    
    def preprocess_sentence_with_features(self, sentence: str) -> list:
        translator = str.maketrans('', '', string.punctuation)
        processed = sentence.split(" ")
        processed = [word.lower().translate(translator) for word in processed]
        processed = self.apply_features(processed)
        return processed
    
    def apply_features(self, sentence):
        
        selected_sentence = sentence
        for i in self.features:
            if i == 'stopwords':
                selected_sentence = self.apply_stopwords(selected_sentence)
            elif i == 'negation':
                selected_sentence = self.apply_negation(selected_sentence)
            elif i == 'binarization':
                selected_sentence = self.apply_binarization(selected_sentence)
            elif i == 'adjectives':
                selected_sentence = self.apply_adjectives(selected_sentence)
            elif i == 'adverbs':
                selected_sentence = self.apply_adverbs(selected_sentence)
            elif i == 'nouns':
                selected_sentence = self.apply_nouns(selected_sentence)
            elif i == 'verbs':
                selected_sentence = self.apply_verbs(selected_sentence)
        
        return selected_sentence
        
    
        