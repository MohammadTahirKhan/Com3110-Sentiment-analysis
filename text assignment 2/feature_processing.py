# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.tokenize import word_tokenize

class FeatureProcesser:
    def __init__(self, data):
        self.features = ['negation', 'binarization', 'POS']
        # self.processed_data_with_features = self.process_data_with_features(self, data)
        self.data = data
        self.NEGATION_TRIGGER_WORDS = ['not', 'no', 'never', 'none', 'nothing', 'neither', 'nor', 'rarley', 'seldom', 'hardly', 'scarcely', 'barely', 'n\'t']
        
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
            if i == 'binarization':
                selected_sentence = self.apply_binarization(selected_sentence)
            # if i == 'POS':
            #     selected_sentence = self.apply_POS(selected_sentence)
            # elif i == 'adjectives':
            #     selected_sentence = self.extract_adjectives(selected_sentence)
            # elif i == 'adverbs':
            #     selected_sentence = self.extract_adverbs(selected_sentence)
            # elif i == 'nouns':
            #     selected_sentence = self.extract_nouns(selected_sentence)
            # elif i == 'verbs':
            #     selected_sentence = self.extract_verbs(selected_sentence)
        
        return selected_sentence
    
    
    def apply_negation(self, sentence: list) -> list:
        negated_sentence = []

        negation_active = False
        for w in sentence:
            # Toggle negation status upon encountering a negation trigger word
            if w in self.NEGATION_TRIGGER_WORDS:
                negation_active = not negation_active
            else:
                # If negation is active, append 'NOT_' to the word
                negated_sentence.append(f'NOT_{w}' if negation_active else w)

        return negated_sentence
        
    def apply_binarization(self, sentence):
        binarized_sentence = []
        for w in sentence:
            if w in binarized_sentence:
                continue
            else:
                binarized_sentence.append(w)
        return binarized_sentence
    
    # def apply_POS(self, sentence):
    #     # Tokenize the sentence and perform part-of-speech tagging
    #     sentence_str = ' '.join(sentence)

    #     # Tokenize the sentence
    #     tokens = word_tokenize(sentence_str)
    #     # tokens = word_tokenize(sentence)
    #     pos_tags = pos_tag(tokens)

    #     # Append the part-of-speech tag to each word
    #     pos_sentence = [f'{word}_{pos}' for word, pos in pos_tags]

    #     return pos_sentence
    
    # def extract_verbs(self, sentence):
    #     tokens = word_tokenize(' '.join(sentence))
    #     pos_tags = pos_tag(tokens)
    #     verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
    #     return verbs

    # def extract_adverbs(self, sentence):
    #     tokens = word_tokenize(' '.join(sentence))
    #     pos_tags = pos_tag(tokens)
    #     adverbs = [word for word, pos in pos_tags if pos.startswith('RB')]
    #     return adverbs

    # def extract_adjectives(self, sentence):
    #     tokens = word_tokenize(' '.join(sentence))
    #     pos_tags = pos_tag(tokens)
    #     adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
    #     return adjectives

    # def extract_nouns(self, sentence):
    #     tokens = word_tokenize(' '.join(sentence))
    #     pos_tags = pos_tag(tokens)
    #     nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
    #     return nouns