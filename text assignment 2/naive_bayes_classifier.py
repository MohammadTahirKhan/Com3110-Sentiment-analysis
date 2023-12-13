import pandas as pd
class NaiveBayesClassifier():
    """
    Naive Bayes model for sentiment analysis
    """
    def __init__(self):
        """
        Initialize Naive Bayes classifier
        """
        self.num_classes = 0
        self.priors = []
        self.feature_likelihoods = {}
        self.vocabulary = set()

    def compute_priors(self, labels):
        """
        Compute priors for each class

        Args:
            labels: list of sentiment labels
        """
        self.num_classes = len(set(labels))
        self.priors = [labels.count(i) / len(labels) for i in range(self.num_classes)]

    def compute_likelihoods(self, data, labels):
        """
        Compute likelihoods for each feature in each class

        Args:
            data: list of phrases
            labels: list of sentiment labels
        """
        self.feature_likelihoods = {}

        for sentiment_class in range(self.num_classes):
            class_data = [data[i] for i in range(len(data)) if labels[i] == sentiment_class]
            self.feature_likelihoods[sentiment_class] = {}

            # count occurrences of each word or feature in class
            for phrase in class_data:
                for word in phrase:
                    # vocabulary is a set of unique words
                    self.vocabulary.add(word)
                    if word not in self.feature_likelihoods[sentiment_class]:
                        self.feature_likelihoods[sentiment_class][word] = 1
                    else:
                        self.feature_likelihoods[sentiment_class][word] += 1
                        
            # find total number of words in class
            total_words_in_class = sum(len(phrase) for phrase in class_data)
            # apply Laplace smoothing
            for word in self.feature_likelihoods[sentiment_class]:
                self.feature_likelihoods[sentiment_class][word] = (self.feature_likelihoods[sentiment_class][word] + 1) / \
                                                                    (total_words_in_class + len(self.vocabulary))
    
    def train(self, data, labels):
        """
        Train Naive Bayes classifier

        Args:
            data: list of phrases
            labels: list of sentiment labels
        """
        self.compute_priors(labels)
        self.compute_likelihoods(data, labels)

    def predict_sentiment(self, phrase):
        """
        Predict sentiment of phrase

        Args:
            phrase: list of words in phrase

        Returns:
            predicted_sentiment: predicted sentiment of phrase
        """
        posterior_probabilities = []

        for sentiment_class in range(self.num_classes):
            posterior = self.priors[sentiment_class]
            
            # calculate posterior probability for each word in phrase
            for word in phrase:
                if word in self.feature_likelihoods[sentiment_class]:
                    likelihood = self.feature_likelihoods[sentiment_class][word]
                else:
                    likelihood = 1 / (len(self.vocabulary) * self.num_classes)
                posterior *= likelihood
                
            posterior_probabilities.append(posterior)
            
        # predict sentiment according to highest posterior probability
        predicted_sentiment = posterior_probabilities.index(max(posterior_probabilities))
        return predicted_sentiment
    
    def save_output(self, user_id, sentence_ids, predicted_labels, dataset_name):
        """
        Save output to file

        Args:
            sentence_ids: list of sentence ids
            predicted_labels: list of predicted labels
            dataset_name: name of dataset
        """
        data = {"SentenceID": sentence_ids, "Sentiment": predicted_labels}
        df = pd.DataFrame(data)

        output_filename = f'{dataset_name}_predictions_{self.num_classes}classes_{user_id}.tsv'
        df.to_csv(output_filename, sep='\t', index=False)