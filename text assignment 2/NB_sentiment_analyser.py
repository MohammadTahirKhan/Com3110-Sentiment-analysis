# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
from data_preprocessor import DataPreprocessor
from naive_bayes import NaiveBayes
from feature_processing import FeatureProcesser
from evaluate import Evaluate

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "aca21mtk" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args


def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """
    training_ids, training_data, training_labels = DataPreprocessor(number_classes).load_and_preprocess(training)
    if features == 'features':
        training_data = FeatureProcesser(training_data).process_data_with_features()
    nb = NaiveBayes()
    nb.train(training_data, training_labels)
    
    dev_ids, dev_data, dev_labels = DataPreprocessor(number_classes).load_and_preprocess(dev)
    if features == 'features':
        dev_data = FeatureProcesser(dev_data).process_data_with_features()
    dev_predicted_labels = []
    for sentence in dev_data:
        dev_predicted_labels.append(nb.predict_sentiment(sentence))
    
    test_ids, test_data, test_labels = DataPreprocessor(number_classes).load_and_preprocess(test)
    if features == 'features':
        test_data = FeatureProcesser(test_data).process_data_with_features()
    test_predicted_labels = []
    for sentence in test_data:
        test_predicted_labels.append(nb.predict_sentiment(sentence))
        
    if output_files:
        Evaluate(number_classes, confusion_matrix, USER_ID).save_predictions(dev_ids, dev_predicted_labels, 'dev')
        Evaluate(number_classes, confusion_matrix, USER_ID).save_predictions(test_ids, test_predicted_labels, 'test')
        
    
    f1_score = Evaluate(number_classes, confusion_matrix, USER_ID).evaluate_performance(dev_predicted_labels, dev_labels)
    
    #You need to change this in order to return your macro-F1 score for the dev set
    # f1_score = 0
    

    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()