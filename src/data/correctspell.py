import nltk
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
import string

nltk.download('words')

word_list = set(words.words())

def correct_word(word):
    if word in word_list:
        return word
    else:
        closest_word = min(word_list, key=lambda w: edit_distance(word, w))
        return closest_word

def spell_checker(sentence):
    words_in_sentence = sentence.split()
    corrected_words = []

    for word in words_in_sentence:
        clean_word = word.strip(string.punctuation)
        corrected_word = correct_word(clean_word.lower())
        if word[-1] in string.punctuation:
            corrected_word += word[-1]
        corrected_words.append(corrected_word)

    return " ".join(corrected_words)

input_sentence = input("Enter a sentence: ")
corrected_sentence = spell_checker(input_sentence)

print(f"Corrected sentence: {corrected_sentence}")
