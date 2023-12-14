# Com3110-Sentiment-analysis
# Installation

## Requirements
- Python 3.9 or higher
- pip

### Python Libraries
- pandas
- nltk
- seaborn
- matplotlib

These are the main libraries used in the project. Dependencies for these can be automatically installed using pip. 'Stopwords', 'averaged_perceptron_tagger', and 'punkt' are also required from the nltk library. These will be automatically downloaded when the program is run. 

## Setup
1. Extract the zip file to a folder of your choice.
2. Open a terminal window in the folder containing the extracted files.
3. Run the following command to install the required libraries:
```shell
pip install -r requirements.txt
```
4. Run the following command to execute the program:
```shell
python NB_sentiment_analyser.py <TRAINING_FILE> <DEV_FILE> <TEST_FILE> -classes <NUMBER_CLASSES> -features <all_words,features> -output_files -confusion_matrix
``` 

where:
1. <TRAINING_FILE> <DEV_FILE> <TEST_FILE> are the paths to the training, dev and
test files, respectively;
2. -classes <NUMBER_CLASSES> should be either 3 or 5, i.e. the number of classes being
predicted;
3. -features is a parameter to define whether you are using your selected features or
no features (i.e. all words);
4. -output_files is an optional value defining whether or not the prediction files should
be saved (default is "files are not saved");
5. -confusion_matrix is an optional value defining whether confusion matrices should
be shown (default is "confusion matrices are not shown").

