import nltk
nltk.download('punkt',  quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
from nltk import pos_tag
from nltk.tokenize import word_tokenize

class FeatureProcessor:
    """
    Feature processor for sentiment analysis
    """
    
    def __init__(self, data):
        """
        Initialize feature processor

        Args:
            data: list of phrases
        """
        self.data = data
        self.NEGATION_TRIGGER_WORDS = ['not', 'no', 'never', 'none', 'nothing', 'neither', 'nor', 'nobody', 'n\'t']
        
    def process_data_with_features(self):
        """
        Process data with features

        Returns:
            processed_data_with_features: list of processed phrases with features
        """
        processed_data_with_features = []
        for phrase in self.data:
            processed_phrase = self.extract_feature(phrase)
            processed_data_with_features.append(processed_phrase)
        return processed_data_with_features
    
    def extract_feature(self, phrase):
        """
        Extract features from phrase

        Args:
            phrase: list of words in phrase

        Returns:
            phrase: list of words in phrase with features
        """
        binarized_phrase = self.apply_binarization(phrase)
        phrase = self.apply_negation(binarized_phrase)
        tokens = word_tokenize(' '.join(binarized_phrase))
        pos_tags = pos_tag(tokens)
        phrase += self.apply_POS(pos_tags)
        return phrase
    
    def apply_negation(self, phrase):
        """
        Apply negation to phrase
        
        Args:
            phrase: list of words in phrase

        Returns:
            negated_phrase: list of words in phrase with negation applied
        """
        negated_phrase = []

        negation_active = False
        for w in phrase:
            # Toggle negation status upon encountering a negation trigger word
            if w in self.NEGATION_TRIGGER_WORDS:
                negation_active = not negation_active
            else:
                # If negation is active, append 'NOT_' to the word
                if negation_active:
                    negated_phrase.append(f'NOT_{w}')
                    negation_active = False
                else:
                    negated_phrase.append(w)

        return negated_phrase
        
    def apply_binarization(self, phrase):
        """
        Apply binarization to phrase

        Args:
            phrase: list of words in phrase

        Returns:
            binarized_phrase: list of words in phrase with binarization applied
        """
        binarized_phrase = []
        for w in phrase:
            if w in binarized_phrase:
                continue
            else:
                binarized_phrase.append(w)
        return binarized_phrase
    
    def apply_POS(self, pos_tags):
        """
        Apply POS, extract adjectives, adverbs, nouns and verbs from phrase

        Args:
            pos_tags: list of POS tags

        Returns:
            pos_tags_words: list of adjectives, adverbs, nouns and verbs in phrase
        """
        adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
        adverbs = [word for (word, tag) in pos_tags if tag.startswith('RB')]
        nouns = [word for (word, tag) in pos_tags if tag.startswith('NN')]
        verbs = [word for (word, tag) in pos_tags if tag.startswith('VB')]
        pos_tags_words = adjectives + adverbs + nouns + verbs
        return pos_tags_words
    
 