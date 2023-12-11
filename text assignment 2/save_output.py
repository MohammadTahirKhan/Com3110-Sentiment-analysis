import pandas as pd

class SaveOutput:
    def __init__(self, number_classes, user_id):
        self.number_classes = number_classes
        self.user_id = user_id

    def save_output(self, sentence_ids: list, predicted_labels: list, dataset_name: str) -> None:
        data = {"SentenceID": sentence_ids, "Sentiment": predicted_labels}
        df = pd.DataFrame(data)

        output_filename = f'{dataset_name}_predictions_{self.number_classes}classes_{self.user_id}.tsv'
        df.to_csv(output_filename, sep='\t', index=False)