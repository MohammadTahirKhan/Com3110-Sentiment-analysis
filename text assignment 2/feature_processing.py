
import string

class FeatureProcesser:
    def __init__(self, data):
        self.features = ['negation', 'binarization', 'adjectives', 'adverbs', 'nouns', 'verbs']
        # self.processed_data_with_features = self.process_data_with_features(self, data)
        self.data = data
        
    def process_data_with_features(self):
        processed_data_with_features = []
        for sentence in self.data:
            processed_sentence = self.preprocess_sentence_with_features(sentence)
            processed_data_with_features.append(processed_sentence)
        return processed_data_with_features
    
    def preprocess_sentence_with_features(self, sentence: str) -> list:
        # translator = str.maketrans('', '', string.punctuation)
        # processed = sentence.split(" ")
        # processed = [word.lower().translate(translator) for word in processed]
        processed = self.extract_feature(sentence)
        return processed
    
    def extract_feature(self, sentence):
        
        selected_sentence = sentence
        for i in self.features:
            if i == 'negation':
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
        
    
    # def apply_negation(self, sentence):
        