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
        for phrase in self.data:
            processed_phrase = self.preprocess_phrase_with_features(phrase)
            processed_data_with_features.append(processed_phrase)
        return processed_data_with_features
    
    def preprocess_phrase_with_features(self, phrase: str) -> list:
        # translator = str.maketrans('', '', string.punctuation)
        # processed = sentence.split(" ")
        # processed = [word.lower().translate(translator) for word in processed]
        processed = self.extract_feature(phrase)
        return processed
    
    def extract_feature(self, phrase):
        
        selected_phrase = phrase
        for i in self.features:
            if i == 'negation':
                selected_phrase = self.apply_negation(selected_phrase)
            if i == 'binarization':
                selected_phrase = self.apply_binarization(selected_phrase)
            # if i == 'POS':
            #     selected_sentence = self.apply_POS(selected_sentence)
            # el
            # if i == 'adjectives':
            #     selected_sentence = self.extract_adjectives(selected_sentence)
            # if i == 'adverbs':
            #     selected_sentence = self.extract_adverbs(selected_sentence)
            # if i == 'nouns':
            #     selected_sentence = self.extract_nouns(selected_sentence)
            # if i == 'verbs':
            #     selected_sentence = self.extract_verbs(selected_sentence)
        
        return selected_phrase
    
    
    def apply_negation(self, phrase: list) -> list:
        negated_phrase = []

        negation_active = False
        for w in phrase:
            # Toggle negation status upon encountering a negation trigger word
            if w in self.NEGATION_TRIGGER_WORDS:
                negation_active = not negation_active
            else:
                # If negation is active, append 'NOT_' to the word
                negated_phrase.append(f'NOT_{w}' if negation_active else w)

        return negated_phrase
        
    def apply_binarization(self, phrase):
        binarized_phrase = []
        for w in phrase:
            if w in binarized_phrase:
                continue
            else:
                binarized_phrase.append(w)
        return binarized_phrase
    
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
    
    def apply_POS(self, sentence):
        tokens = word_tokenize(' '.join(sentence))  # Convert the list of words to a string
        pos_tags = pos_tag(tokens)
        # Extract POS tags and append them to the sentence
        pos_tags_only = [tag for word, tag in pos_tags]
        sentence_with_pos = sentence + [' '.join(pos_tags_only)]
        return sentence_with_pos
    

    def extract_adjectives(self, sentence):
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        adjectives = [word for word, pos in tagged_words if pos.startswith('JJ')]
        return adjectives

    def extract_adverbs(self, sentence):
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        adverbs = [word for (word, tag) in pos_tags if tag.startswith('RB') and word.lower() not in self.stop_words]
        return adverbs

    def extract_nouns(self, sentence):
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        nouns = [word for (word, tag) in pos_tags if tag.startswith('NN') and word.lower() not in self.stop_words]
        return nouns

    def extract_verbs(self, sentence):
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        verbs = [word for (word, tag) in pos_tags if tag.startswith('VB') and word.lower() not in self.stop_words]
        return verbs