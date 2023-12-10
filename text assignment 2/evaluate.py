import seaborn as sns
import matplotlib.pyplot as plt

class Evaluate:
    def __init__(self, number_classes, confusion_matrix, user_id):
        self.number_classes = number_classes
        self.confusion_matrix = confusion_matrix
        self.user_id = user_id

    def initialize_confusion_matrix(self):
        return [[0 for _ in range(self.number_classes)] for _ in range(self.number_classes)]

    def print_confusion_matrix(self, confusion_matrix_counts):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_counts, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title("Confusion Matrix")
        plt.show()
        # print("Confusion matrix:")
        # for row in confusion_matrix_counts:
        #     print(row)

    def compute_f1_scores(self, confusion_matrix_counts):
        f1_scores = []
        for class_label in range(self.number_classes):
            tp = confusion_matrix_counts[class_label][class_label]

            other_classes = [c for c in range(self.number_classes) if c != class_label]
            fps = [confusion_matrix_counts[other_class_label][class_label] for other_class_label in other_classes]
            fns = [confusion_matrix_counts[class_label][other_class_label] for other_class_label in other_classes]

            f1_scores.append(2 * tp / (2 * tp + sum(fps) + sum(fns)))

        return f1_scores

    def evaluate_performance(self, predicted_labels: list, actual_labels: list) -> float:
        confusion_matrix_counts = self.initialize_confusion_matrix()
        correct = 0

        for i in range(len(actual_labels)):
            if predicted_labels[i] == actual_labels[i]:
                correct += 1

            confusion_matrix_counts[actual_labels[i]][predicted_labels[i]] += 1

        if self.confusion_matrix:
            self.print_confusion_matrix(confusion_matrix_counts)

        f1_scores = self.compute_f1_scores(confusion_matrix_counts)
        return sum(f1_scores) / len(f1_scores)

    def save_predictions(self, sentence_ids: list, predicted_labels: list, dataset_name: str) -> None:
        lines_to_write = ["SentenceID\tSentiment\n"]
        lines_to_write.extend([f"{sentence_ids[i]}\t{predicted_labels[i]}\n" for i in range(len(sentence_ids))])

        output_filename = f'{dataset_name}_predictions_{self.number_classes}classes_{self.user_id}.tsv'
        with open(output_filename, 'w') as f:
            f.writelines(lines_to_write)
