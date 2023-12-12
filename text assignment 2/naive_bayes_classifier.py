class NaiveBayesClassifier():
    """
    Naive Bayes model for sentiment analysis
    """
    def __init__(self):
        self.num_classes = 0
        self.priors = []
        self.feature_likelihoods = {}
        self.vocabulary = set()

    def compute_priors(self, labels):
        self.num_classes = len(set(labels))
        self.priors = [labels.count(i) / len(labels) for i in range(self.num_classes)]

    def compute_likelihoods(self, data, labels):
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
        self.compute_priors(labels)
        self.compute_likelihoods(data, labels)

    def predict_sentiment(self, phrase):
        posterior_probabilities = []

        for sentiment_class in range(self.num_classes):
            posterior = self.priors[sentiment_class]
            
            for word in phrase:
                if word in self.feature_likelihoods[sentiment_class]:
                    likelihood = self.feature_likelihoods[sentiment_class][word]
                else:
                    likelihood = 1 / (len(self.vocabulary) * self.num_classes)
                posterior *= likelihood
                
            posterior_probabilities.append(posterior)
            
        predicted_sentiment = posterior_probabilities.index(max(posterior_probabilities))
        return predicted_sentiment

                
                
                
        #         likelihood = self.feature_likelihoods[sentiment_class].get(word, 1 / (len(self.vocabulary) * self.num_classes))
        #         posterior *= likelihood

        #     posterior_probabilities.append(posterior)

        # predicted_sentiment = posterior_probabilities.index(max(posterior_probabilities))
        # return predicted_sentiment