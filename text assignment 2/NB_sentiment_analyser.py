# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
from data_preprocessor import DataPreprocessor
from naive_bayes_classifier import NaiveBayesClassifier
from feature_processing import FeatureProcessor
from evaluate import Evaluate

USER_ID = "aca21mtk" #unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

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
    
    data_preprocessor = DataPreprocessor(number_classes)
    training_ids, training_data, training_labels = data_preprocessor.preprocess(training)
    if features == 'features':
        training_data = FeatureProcessor(training_data).process_data_with_features()
    classifier = NaiveBayesClassifier()
    classifier.train(training_data, training_labels)
    
    dev_ids, dev_data, dev_labels = data_preprocessor.preprocess(dev)
    if features == 'features':
        dev_data = FeatureProcessor(dev_data).process_data_with_features()
    dev_predicted_labels = []
    for phrase in dev_data:
        dev_predicted_labels.append(classifier.predict_sentiment(phrase))
    
    test_ids, test_data, test_labels = data_preprocessor.preprocess(test)
    if features == 'features':
        test_data = FeatureProcessor(test_data).process_data_with_features()
    test_predicted_labels = []
    for phrase in test_data:
        test_predicted_labels.append(classifier.predict_sentiment(phrase))
        
    if output_files:
        # save output to files
        classifier.save_output(USER_ID, dev_ids, dev_predicted_labels, 'dev')
        classifier.save_output(USER_ID, test_ids, test_predicted_labels, 'test')
        
    evaluator = Evaluate(number_classes, confusion_matrix, USER_ID)
    f1_score = evaluator.evaluate_performance(dev_predicted_labels, dev_labels)
    
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))
    
    if confusion_matrix:
        evaluator.print_confusion_matrix()

if __name__ == "__main__":
    main()