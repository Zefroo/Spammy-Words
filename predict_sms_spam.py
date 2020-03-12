# -*- coding: utf-8 -*-
import csv
import re
import math
import random
from scipy import stats
import numpy as np
import contractions
import string
from collections import Counter
import enchant

#-------Below is the program to generate a .csv file to test the accuracy of features--------------------#

def correct_words(word_list):
    """Takes a list of strings and tries to correct them so that they are valid words (i.e. alphabetical).
    Correction Procedure: (1) make lowercase -> (2) delete string if not alphabetical (allowed to contain an
    apostrophe or punctuation at the end) -> (3) remove trailing punctuation -> (4) break up contractions,
    split and take original word -> (5) remove possessive 's -> (6) check if word is in dictionary, if not,
    attempt to correct it to most likely correct word, and if none can be found, delete it.
    Returns list of corrected words"""
    english_dict = enchant.request_dict("en_US")  # open up the english dictionary
    index = 0
    while index <= len(word_list) - 1:
        # make lowercase
        word_list[index] = word_list[index].lower()
        # remove values if they aren't words (alphabetical, can end with punctuation)
        if not re.match('^[a-z]+[?,;.!]*$', word_list[index]):
            del word_list[index]
            index -= 1
        # it is a word - clean it up.
        else:
            # remove punctuation if it appears at the end of a word.
            word_list[index] = word_list[index].rstrip(string.punctuation)
            # break up contractions using contraction library (chooses most likely conversion - can be mistaken
            word_list[index] = contractions.fix(word_list[index])
            # split and remove contraction
            word_list[index] = word_list[index].split()[0]
            # remove "'s" at the ends of words (contraction library doesn't remove possessive 's)
            word_list[index] = re.sub("\'s$", '', word_list[index])
            # spell check words
            if not english_dict.check(word_list[index]):
                suggestions = english_dict.suggest(word_list[index])
                if len(suggestions) > 0:
                    word_list[index] = suggestions[0].lower().split()[0]
                else:
                    del word_list[index]
                    index -= 1
        index += 1
        for item in word_list:
            if not item.isalpha():
                word_list.remove(item)
    return word_list


def get_unique_words():
    """Return a dictionary of unique words from a list of texts (where texts are sub-lists)
    where the key is the unique word and the value is the number of times that word appears in texts"""
    csv_file = open('spam.csv', 'r')
    csv_reader = csv.reader(csv_file, delimiter=",")
    words_dict = dict()
    texts_counter = 0  # count how many texts there are in total
    for line in csv_reader:
        text = line[1]
        texts_counter += 1
        words_in_text = text.split()  # creates a list of all the words in a text
        corrected_text = correct_words(words_in_text)
        unique_words = set(corrected_text)
        for word in unique_words:
            words_dict[word] = words_dict.get(word, 0) + 1
    return words_dict, texts_counter


def does_have_links(text):
    """Returns True if a link is present in the text. Else returns False"""
    link = "http"
    if link in text:
        return "1"
    else:
        return "0"


def does_have_spam_words(text):
    """writes whether the line contains spam words into the csv file
    in the second column"""
    # Spam word list based on findings from original spam.csv file
    spam_words = ['call', 'winner', 'free', 'mobile', 'claim', 'text', 'reply', 'contact',
                  'now', 'send', 'prize', 'won', 'win', 'service', 'cash', 'urgent',
                  'award', 'reward', 'draw', 'receive', 'customer', 'entry', 'enter', 'selected',
                  'guarantee', 'guaranteed', 'valid', 'chance', 'trip', 'redeem', 'awarded', 'delivery']
    words_in_text = text.split()
    corrected_words = correct_words(words_in_text)
    count = 0
    for word in corrected_words:
        if word in spam_words:
            count += 1
    return str(count)


def percent_non_alpha(text):
    """Calculates percentage of non alphabetical and space characters.
     Based on findings from spam.csv that spam texts have, on average, significantly more
     non alphabetical characters than ham  texts (see: avg_non_alpha())"""
    char_count = len(text)
    non_alpha_count = 0
    for char in text:
        if not char.isalpha() and not char.isspace():
            non_alpha_count += 1
    perc_non_alpha = float(non_alpha_count) / float(char_count)
    return str(perc_non_alpha)


def percent_uppercase(text):
    """Calculates percentage of alphabetical characters that are uppercase, out of total alphabetical characters.
    Based on findings from spam.csv that spam texts have higher uppercase alphabetical characters
    (see: avg_uppercase_letters())"""
    alpha_count = 0
    uppercase_count = 0
    for char in text:
        if char.isalpha():
            alpha_count += 1
            if char.isupper():
                uppercase_count += 1
    # calculate percentage - make sure not to divide by 0
    try:
        perc_uppercase = float(uppercase_count) / float(alpha_count)
        return str(perc_uppercase)
    except ZeroDivisionError:
        return "0"


def count_vector(text):
    """Creates a vector of unique words and how many times they appear."""
    count_dict = Counter(text)
    return count_dict


def term_frequency(text, word):
    """Takes in a text (list of strings) and calculates
    TF(t) = (number of times term appears in text) / (total number of terms in a text)"""
    total_words = len(text)
    word_in_text = text.count(word)
    try:
        return float(word_in_text) / float(total_words)
    except ZeroDivisionError:
        return 0


def inverse_document_frequency(word_occurrence, num_texts):
    """Takes in a word (string) and texts (list of lists and calculates the
    number of texts over number of texts where the word occurs"""
    try:
        IDF = float(num_texts) / float(word_occurrence)
        return math.log(IDF)
    except ZeroDivisionError:
        return 0


def tf_idf_vector(term_frequency, idf):
    """Calculates tf/idf vector"""
    try:
        return term_frequency * idf
    except ZeroDivisionError:
        return 0


def write_new_file(word_dictionary):
    """Creates the new test file with appropriate headers for the spam and ham data analysis
    for all texts - with count vectors"""
    f = open("testFile.csv", "w+")
    f.write("class_feature, has_link, spammy_words, perc_non_alpha, perc_upper,")

    column_index = 0
    # key in unique words from word_dictionary
    for key, value in word_dictionary.items():
        if column_index == len(word_dictionary) - 1:
            f.write(key + "CV")
        else:
            f.write(key + "CV,")
            column_index += 1
    column_index = 0
    for key, value in word_dictionary.items():
        if column_index == len(word_dictionary) - 1:
            f.write(key + "IDF")
        else:
            f.write(key + "IDF,")
            column_index += 1
    f.write("\n")
    return f


def main():
    """create the spam feature file"""
    csv_file = open('spam.csv', "r")
    next(csv_file)
    read_lines = csv.reader(csv_file)
    # get data from file
    extract_file = get_unique_words()
    word_dictionary = extract_file[0]
    texts_count = extract_file[1]

    # create a file with count vectors
    f = write_new_file(word_dictionary)
    for text in read_lines:
        words_in_text_message = text[1].split(' ')
        cleaned_text = correct_words(words_in_text_message)
        # add ham or spam
        f.write(text[0] + ",")
        # add whether it has links
        has_link = does_have_links(text[1])
        f.write(has_link + ",")
        # add spammy words count
        spammy_words = does_have_spam_words(text[1])
        f.write(spammy_words + ",")
        # add percent non alpha
        non_alpha = percent_non_alpha(text[1])
        f.write(non_alpha + ",")
        # add percent uppercase
        upper = percent_uppercase(text[1])
        f.write(upper + ",")
        # add count vector to file
        index = 0
        count_vec = count_vector(cleaned_text)
        for key, value in word_dictionary.items():
            if key in words_in_text_message:
                if index == len(words_in_text_message) - 1:
                    f.write(str(count_vec[key]))
                else:
                    f.write(str(count_vec[key]) + ",")
            else:
                if index == len(words_in_text_message) - 1:
                    f.write("0")
                else:
                    f.write("0,")
            index += 1
        index = 0
        # add tf/idf vectors to file
        for key, value in word_dictionary.items():
            word_occurrence = word_dictionary.get(key, 0)
            term_freq = term_frequency(cleaned_text, key)
            idf_vec = inverse_document_frequency(word_occurrence, texts_count)
            tf_idf = tf_idf_vector(term_freq, idf_vec)
            if key in words_in_text_message:
                if index == len(words_in_text_message) - 1:
                    f.write(str(tf_idf))
                else:
                    f.write(str(tf_idf) + ",")
            else:
                if index == len(words_in_text_message) - 1:
                    f.write("0")
                else:
                    f.write("0,")
            index += 1
        f.write("\n")


# main()


#---------------------------The following code is for the second program to predict spam-------------------------#
# use functions above: does_have_spam(text), percent_non_alpha(text), percent_uppercase(text)

def get_prediction(spam_count, uppercase, non_alpha):
    """Based on tree visualizer from Weka.
    I have included elif statements instead of else for clarity purposes (made it easier to record everything)"""
    if spam_count <= 1:
        if spam_count == 0:
            return "ham"
        elif spam_count == 1:
            if non_alpha <= 0.10274:
                return "ham"
            elif non_alpha > 0.10274:
                if uppercase > 0.06666667:
                    return "spam"
                elif uppercase <= 0.06666667:
                    if non_alpha <= 0.146341:
                        return "ham"
                    elif non_alpha > 0.146341:
                        return "spam"
    else: # spam_count > 1
        if non_alpha <= 0.089888:
            if uppercase <= 0.047059:
                if spam_count <= 2:
                    return "ham"
                elif spam_count > 2:
                    if uppercase > 0.027523:
                        return "ham"
                    elif uppercase <= 0.027523:
                        return "spam"
            elif uppercase > 0.047059:
                if non_alpha <= 0.050505:
                    if spam_count <= 2:
                        if non_alpha <= 0.034722:
                            return "spam"
                        elif non_alpha > 0.034722:
                            return "ham"
                    elif spam_count > 2:
                        if uppercase <= 0.119048:
                            return "ham"
                        elif uppercase > 0.119048:
                            return "spam"
                elif non_alpha > 0.050505:
                    if uppercase > 0.268293:
                        return "ham"
                    elif uppercase <= 0.268293:
                        if uppercase > 0.113402:
                            return "spam"
                        elif uppercase <= 0.113402:
                            if uppercase > 0.106557:
                                return "ham"
                            elif uppercase <= 0.106557:
                                return "spam"
        elif non_alpha > 0.089888:
            if uppercase > 0.065574:
                return "spam"
            elif uppercase <= 0.065574:
                if spam_count > 2:
                    return "spam"
                elif spam_count <= 2:
                    if uppercase > 0.061538:
                        return "ham"
                    elif uppercase <= 0.061538:
                        if non_alpha > 0.123288:
                            return "spam"
                        elif non_alpha <= 0.123288:
                            return "ham"


def predict_spam():
    text_message = raw_input("Please enter text: ")
    #compute features
    spam_count = float(does_have_spam_words(text_message))
    uppercase = float(percent_uppercase(text_message))
    non_alpha = float(percent_non_alpha(text_message))

    prediction = get_prediction(spam_count, uppercase, non_alpha)
    print prediction

predict_spam()

# --------------------------The following code was just for testing the features I developed --------------------#
def test_number_words_appear_for_spam_vs_ham():
    """find counts for words that appear in spam and ham texts.
    Used to determine which words are 'spammy' or not"""
    csv_file = open('spam.csv', 'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    spam_words = list()
    ham_words = list()
    for row in csv_reader:
        if row[0] == "ham":
            ham_line = row[1].split()
            for item in ham_line:
                ham_words.append(item)
        else:
            spam_line = row[1].split()
            for item in spam_line:
                spam_words.append(item)
    ham_words_dict = Counter(ham_words)
    spam_words_dict = Counter(spam_words)
    csv_file.close()

    # create spam word file
    spam_csv = open('count_spam_items.csv', 'wb')
    spam_writer = csv.writer(spam_csv)
    spam_writer.writerow(['word', 'count'])
    spam_writer.writerows(spam_words_dict.items())

    # create ham word file
    ham_csv = open('count_ham_items.csv', 'wb')
    ham_writer = csv.writer(ham_csv)
    ham_writer.writerow(['word', 'count'])
    ham_writer.writerows(ham_words_dict.items())

    ham_csv.close()
    spam_csv.close()


def avg_not_alpha():
    """Calculates how common a text has non alphabetical characters
    on average in spam vs. ham texts
    Tests whether the difference between spam vs. ham is significant"""
    csv_file = open('spam.csv', 'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    spam_non_alpha = []
    ham_non_alpha = []
    for row in csv_reader:
        if row[0] == 'spam':
            total_chars = 0
            non_alpha_counter = 0
            for char in row[1]:
                total_chars += 1
                if not char.isalpha() and not char.isspace():
                    non_alpha_counter += 1
            spam_non_alpha.append(float(non_alpha_counter) / float(total_chars))
        else:  # row[0] == ham
            total_chars = 0
            non_alpha_counter = 0
            for char in row[1]:
                total_chars += 1
                if not char.isalpha() and not char.isspace():
                    non_alpha_counter += 1
            ham_non_alpha.append(float(non_alpha_counter) / float(total_chars))
    # much more ham data than spam, so take out same number of spam data to sample it
    ham_sample = random.sample(ham_non_alpha, 747)

    # test whether it is statistically significant
    print(stats.ttest_ind(ham_sample, spam_non_alpha))
    print("The avg % of non alphabetical characters in ham is: " + str(np.mean(ham_non_alpha)))
    print("The avg % of non alphabetical characters in spam is: " + str(np.mean(spam_non_alpha)))
    csv_file.close()


def avg_uppercase_letter():
    """Calculates how common uppercase letters appear out of total alphabetical characters
    in text on average in spam vs. ham texts.
    Tests whether the difference between spam vs. ham is significant"""
    csv_file = open('spam.csv', 'r')
    csv_reader = csv.reader(csv_file, delimiter=",")
    spam_uppercase = []
    ham_uppercase = []
    for row in csv_reader:
        if row[0] == 'spam':
            total_alpha = 0
            upper_counter = 0
            for char in row[1]:
                if char.isalpha():
                    total_alpha += 1
                if char.isupper():
                    upper_counter += 1
            try:
                spam_uppercase.append(float(upper_counter) / float(total_alpha))
            # if there are 0 alphabet characters in the text, ignore.
            except ZeroDivisionError:
                continue
        else:  # row[0] == ham
            total_alpha = 0
            upper_counter = 0
            for char in row[1]:
                if char.isalpha():
                    total_alpha += 1
                if char.isupper():
                    upper_counter += 1
            try:
                ham_uppercase.append(float(upper_counter) / float(total_alpha))
            # if there are 0 alphabet characters in the text, ignore.
            except ZeroDivisionError:
                continue
    # much more ham data than spam, so take out same number of spam data to sample it
    ham_sample = random.sample(ham_uppercase, 747)

    # test whether it is statistically significant
    print(stats.ttest_ind(ham_sample, spam_uppercase))
    print("The average % of uppercase values in ham is: " + str(np.mean(ham_uppercase)))
    print("The avg % of uppercase values in spam is: " + str(np.mean(spam_uppercase)))
    csv_file.close()

# test_number_words_appear_for_spam_vs_ham()
# avg_not_alpha()
# avg_uppercase_letter()

