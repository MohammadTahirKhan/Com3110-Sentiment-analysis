# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.tokenize import word_tokenize

class FeatureProcesser:
    def __init__(self, data):
        self.features = ['negation', 'binarization', 'POS']
        self.data = data
        self.NEGATION_TRIGGER_WORDS = ['not', 'no', 'never', 'none', 'nothing', 'neither', 'nor', 'rarley', 'seldom', 'hardly', 'scarcely', 'barely', 'n\'t']
        
    def process_data_with_features(self):
        processed_data_with_features = []
        for phrase in self.data:
            processed_phrase = self.preprocess_phrase_with_features(phrase)
            processed_data_with_features.append(processed_phrase)
        return processed_data_with_features
    
    def preprocess_phrase_with_features(self, phrase: str) -> list:
        processed = self.extract_feature(phrase)
        return processed
    
    def extract_feature(self, phrase):
        selected_phrase = phrase
        # tokens = word_tokenize(' '.join(selected_phrase))
        # pos_tags = pos_tag(tokens)
        
        for i in self.features:
            # if i == 'negation':
            #     selected_phrase = self.apply_negation(selected_phrase)
            if i == 'POS':
                tokens = word_tokenize(' '.join(selected_phrase))
                pos_tags = pos_tag(tokens)
                selected_phrase += self.apply_POS(pos_tags)
            # if i == 'adjectives':
            #     selected_phrase += self.extract_adjectives(pos_tags)
            # if i == 'adverbs':
            #     selected_phrase += self.extract_adverbs(pos_tags)
            # if i == 'nouns':
            #     selected_phrase += self.extract_nouns(pos_tags)
            # if i == 'verbs':
            #     selected_phrase += self.extract_verbs(pos_tags)
            if i == 'binarization':
                selected_phrase = self.apply_binarization(selected_phrase)
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
    
    # def apply_POS(self, pos_tags):
    #     # Extract POS tags and append them to the sentence
    #     pos_tags_only = [tag for word, tag in pos_tags]
    #     sentence_with_pos = sentence + [' '.join(pos_tags_only)]
    #     return sentence_with_pos
    
    def apply_POS(self, pos_tags):
        # tokens = word_tokenize(' '.join(sentence))
        # pos_tags = pos_tag(tokens)
        adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
        adverbs = [word for (word, tag) in pos_tags if tag.startswith('RB')]
        nouns = [word for (word, tag) in pos_tags if tag.startswith('NN')]
        verbs = [word for (word, tag) in pos_tags if tag.startswith('VB')]
        pos_tags_words = adjectives + adverbs + nouns + verbs
        return pos_tags_words
    
    # def extract_adjectives(self, pos_tags):
    #     # words = word_tokenize(' '.join(sentence))
    #     # tagged_words = pos_tag(words)
    #     adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
    #     return adjectives

    # def extract_adverbs(self, pos_tags):
    #     # tokens = word_tokenize(' '.join(sentence))
    #     # pos_tags = pos_tag(tokens)
    #     adverbs = [word for (word, tag) in pos_tags if tag.startswith('RB')]
    #     return adverbs

    # def extract_nouns(self, pos_tags):
    #     # tokens = word_tokenize(' '.join(sentence))
    #     # pos_tags = pos_tag(tokens)
    #     nouns = [word for (word, tag) in pos_tags if tag.startswith('NN')]
    #     return nouns

    # def extract_verbs(self, pos_tags):
    #     # tokens = word_tokenize(' '.join(sentence))
    #     # pos_tags = pos_tag(tokens)
    #     verbs = [word for (word, tag) in pos_tags if tag.startswith('VB')]
    #     return verbs