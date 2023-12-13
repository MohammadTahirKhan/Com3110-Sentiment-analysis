import seaborn as sns
import matplotlib.pyplot as plt

class Evaluate:
    """ 
    Evaluate the performance of the model by computing the F1 score for each class and averaging them.
    """
    
    def __init__(self, number_classes, confusion_matrix, user_id):
        """
        Initialize evaluator.

        Args:
            number_classes: number of classes
            confusion_matrix: whether to print confusion matrix
            user_id: student user id
        """
        self.number_classes = number_classes
        self.confusion_matrix = confusion_matrix
        self.user_id = user_id
        self.confusion_matrix_counts = []

    def initialize_confusion_matrix(self):
        """
        Initialize confusion matrix with zeros.

        Returns:
            confusion matrix containing zeros
        """
        return [[0 for _ in range(self.number_classes)] for _ in range(self.number_classes)]

    def print_confusion_matrix(self):
        """
        Print confusion matrix as a heatmap.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.confusion_matrix_counts, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title("Confusion Matrix")
        plt.show()
    
    def compute_f1_scores(self, confusion_matrix_counts):
        """ 
        Compute the F1 score for each class

        Args:
            confusion_matrix_counts: confusion matrix containing the count for correct
                                      and incorrect predictions for each class

        Returns:
            f1_scores: list of F1 scores for each class
        """
        f1_scores = []
        for class_label in range(self.number_classes):
            tp = confusion_matrix_counts[class_label][class_label]

            other_classes = [c for c in range(self.number_classes) if c != class_label]
            fp = 0
            fn = 0
            for other_class_label in other_classes:
                fp += confusion_matrix_counts[other_class_label][class_label]
                fn += confusion_matrix_counts[class_label][other_class_label]

            f1_scores.append(2 * tp / (2 * tp + fp + fn))
        return f1_scores

    def evaluate_performance(self, predicted_labels, actual_labels):
        """
        Evaluate the performance of the model by computing the F1 score for each class and averaging them.

        Args:
            predicted_labels: list of predicted labels
            actual_labels: list of actual labels

        Returns:
            average F1 score for all classes
        """
        self.confusion_matrix_counts = self.initialize_confusion_matrix()
        correct = 0

        for i in range(len(actual_labels)):
            if predicted_labels[i] == actual_labels[i]:
                correct += 1
            self.confusion_matrix_counts[actual_labels[i]][predicted_labels[i]] += 1

        f1_scores = self.compute_f1_scores(self.confusion_matrix_counts)
        return sum(f1_scores) / len(f1_scores)