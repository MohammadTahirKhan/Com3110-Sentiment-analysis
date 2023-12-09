class NaiveBayes():
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
        self.priors = [labels.count(c) / len(labels) for c in range(self.num_classes)]

    def compute_likelihoods(self, data, labels):
        self.feature_likelihoods = {}

        for sentiment_class in range(self.num_classes):
            class_samples = [data[i] for i in range(len(data)) if labels[i] == sentiment_class]
            self.feature_likelihoods[sentiment_class] = {}

            for sample in class_samples:
                for feature in sample:
                    self.vocabulary.add(feature)
                    if feature not in self.feature_likelihoods[sentiment_class]:
                        self.feature_likelihoods[sentiment_class][feature] = 1
                    else:
                        self.feature_likelihoods[sentiment_class][feature] += 1

            total_samples = sum(len(sample) for sample in class_samples)
            for feature in self.feature_likelihoods[sentiment_class]:
                self.feature_likelihoods[sentiment_class][feature] = (self.feature_likelihoods[sentiment_class][feature] + 1) / \
                                                                     (total_samples + len(self.vocabulary))
    
    def train(self, data, labels):
        self.compute_priors(labels)
        self.compute_likelihoods(data, labels)

    def predict_sentiment(self, data):
        posterior_probabilities = []

        for sentiment_class in range(self.num_classes):
            posterior = self.priors[sentiment_class]

            for feature in data:
                likelihood = self.feature_likelihoods[sentiment_class].get(feature, 1 / (len(self.vocabulary) * self.num_classes))
                posterior *= likelihood

            posterior_probabilities.append(posterior)

        predicted_sentiment = posterior_probabilities.index(max(posterior_probabilities))
        return predicted_sentiment