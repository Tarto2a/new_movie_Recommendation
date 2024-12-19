import nltk
import pandas as pd
import os
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from spellchecker import SpellChecker
from googletrans import Translator
from langdetect import detect

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
translator = Translator()
spell = SpellChecker()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOPWORDS and word not in string.punctuation
    ]
    return " ".join(tokens)

def correct_spelling(user_input):
    words = user_input.split()
    corrected_words = [spell.correction(word) for word in words]
    return " ".join(corrected_words)

def expand_with_synonyms(user_input):
    words = user_input.split()
    expanded_terms = set(words)
    for word in words:
        synonyms = wordnet.synsets(word)
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_terms.add(preprocess_text(lemma.name().replace('_', ' ')))
    return ' '.join(expanded_terms)

def detect_and_translate(input_text):
    try:
        detected_language = detect(input_text)
        if detected_language != 'en':
            translated_text = translator.translate(input_text, src=detected_language, dest='en').text
            return translated_text
        return input_text
    except Exception as e:
        print(f"Error in translation: {e}")
        return input_text

if __name__ == "__main__":
    input_file = os.path.join("data", "raw", "movies.csv")
    output_file = os.path.join("data", "processed", "processed_movies.csv")

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
    else:
        df = pd.read_csv(input_file)

        if 'description' not in df.columns:
            print(f"Error: 'description' column not found in '{input_file}'.")
        else:
            df['processed_description'] = df['description'].apply(preprocess_text)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"Processed data has been saved to '{output_file}'.")
