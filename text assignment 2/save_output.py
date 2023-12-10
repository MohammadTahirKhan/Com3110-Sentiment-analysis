class SaveOutput:
    def __init__(self, number_classes, user_id):
        self.number_classes = number_classes
        self.user_id = user_id

    def save_output(self, sentence_ids: list, predicted_labels: list, dataset_name: str) -> None:
        lines_to_write = ["SentenceID\tSentiment\n"]
        lines_to_write.extend([f"{sentence_ids[i]}\t{predicted_labels[i]}\n" for i in range(len(sentence_ids))])

        output_filename = f'{dataset_name}_predictions_{self.number_classes}classes_{self.user_id}.tsv'
        with open(output_filename, 'w') as f:
            f.writelines(lines_to_write)
