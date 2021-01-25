import re

def void_term(x, word_list, print_word=False, return_bool=False):
    c_list = []
    breaker = False
    term_present = False
    for sentence in x:
        for word in word_list:
            if word in sentence:
                if print_word:
                    print("word is: ", word, "in sentence: ", sentence)
                term_present = True
                breaker = True
                break
        if not breaker:
            c_list.append(sentence.strip())
        breaker = False
    if return_bool:
        return not term_present
    else:
        return c_list

def validate_term(x, word_list, print_word=False, return_bool=False):
    c_list = []
    term_present = False
    for sentence in x:
        for word in word_list:
            if word in sentence:
                c_list.append(sentence.strip())
                term_present = True
                if print_word:
                    print("word is: ", word, "in sentence: ", sentence)
                break
    if return_bool:
        return term_present
    else:
        return c_list

def regex_breaker(x, regex, return_bool=True):
    c_list = []
    term_present = False
    for sentence in x:
        if re.search(regex, sentence):
            term_present = True
        else:
            c_list.append(sentence)

    if return_bool:
        return not term_present
    else:
        return c_list