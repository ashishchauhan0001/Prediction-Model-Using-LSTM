import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def clean_and_save_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    cleaned_data = []

    for sentence in data:
        # Lowercase all words
        sentence = sentence.lower()

        # Handle contractions (e.g., "don't" becomes "do n't")
        sentence = re.sub(r"n't", " n't", sentence)
        sentence = re.sub(r"'s", " 's", sentence)
        sentence = re.sub(r"'re", " 're", sentence)
        sentence = re.sub(r"'m", " 'm", sentence)
        sentence = re.sub(r"'ll", " 'll", sentence)
        sentence = re.sub(r"'d", " 'd", sentence)
        sentence = re.sub(r"'ve", " 've", sentence)

        # Define a regular expression pattern to keep only English words and punctuation marks
        pattern = re.compile(r"[A-Za-z.,?!']+")

        # Apply the pattern to each sentence and join the words to form cleaned sentences
        cleaned_sentence = ' '.join(re.findall(pattern, sentence))

        # Tokenize the cleaned sentence using NLTK
        tokens = word_tokenize(cleaned_sentence)

        # Join the tokens back into a sentence
        cleaned_data.append(' '.join(tokens))

    # Save the modified data back to the same file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(cleaned_data)

# Clean and save the training data
clean_and_save_data('train.txt')

# Clean and save the test data
clean_and_save_data('test.txt')