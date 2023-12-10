import csv
import string
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

class DataPreprocessor:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.CLASS_MAPPING = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
        self.current_id = None
        # self.features = features
        self.stop_words = stopwords.words('english')
        self.stop_words.extend([ '\'s', '\'d', '\'ll', '\'re', '\'ve', 'n\'t', '``', '\'\'', '...', '—', '’', '“', '”', '‘', '...', '--' ])
        self.stop_words.extend(string.punctuation.replace('!', ''))

    def preprocess(self, sentence: str) -> list:
        # processed = sentence.split(" ")
        # processed = [word.lower() for word in processed]
        # return processed
        # translator = str.maketrans('', '', string.punctuation)
        processed = sentence.split(" ")
        # processed = [word.lower().translate(translator) for word in processed]
        
        # stop_words = stopwords.words('english')
        # stop_words.extend([ '\'s', '\'d', '\'ll', '\'re', '\'ve', 'n\'t', '``', '\'\'', '...', '—', '’', '“', '”', '‘', '...', '--' ])
        # stop_words.extend(string.punctuation.replace('!', ''))
        
        processed = [word.lower() for word in processed if word.lower() not in self.stop_words]
        
        return processed
    
    # def preprocess_sentence_with_features(self, sentence: str) -> list:
    #     processed = sentence.split(" ")
    #     processed = [word.lower() for word in processed]
        
    #     # self.PUNCTUATIONS = ['.', ',', '?', '!', ':', ';', '-lrb-', '-rrb-', '-lsb-', '-rsb-', '-lcb-', '-rcb-', '-', '--', '_', '`', '``', '\'', '\'\'', '...', '—']
        
        
    #     return processed
    

    def load_and_preprocess(self, filename: str) -> tuple:
        ids = []  # sentence ids
        data = []  # sentences
        labels = []  # sentiments

        with open(filename) as f:
            read_data = csv.reader(f, delimiter='\t')
            next(read_data, None)  # skip column headings and ignore return value
            for line in read_data:
                self.current_id = line[0]
                sentence = line[1]
                
                # if self.features == 'all_words':
                processed_sentence = self.preprocess(sentence)
                # else:
                #     processed_sentence = self.preprocess_sentence_with_features(sentence)
                
                ids.append(self.current_id)
                data.append(processed_sentence)

                if filename != 'moviereviews/test.tsv':
                    if self.num_classes == 5:
                        sentiment_label = int(line[2])
                    else:
                        sentiment_label = self.CLASS_MAPPING[int(line[2])]
                    labels.append(sentiment_label)

        return ids, data, labels