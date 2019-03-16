#! /usr/bin/env python3
"""compute some standard information retrieval metrics, tf, idf and tf-idf"""


def tf(term, document):
    """
    computes term frequency. TF is defined as how often the term in question
    appears in a document over the sum of all terms in the document:
    (term/all_terms_in_doc).

    Parameters:
        term: a string containing the search term
        document: a list representing the document text, split into tokens
            (make sure that punctuation is split, too!)

    Return Value: a float representing the tf value
    """
    terms_found = 0
    for token in document:
        if token == term:
            terms_found += 1
    return terms_found/len(document)


def idf(term, corpus):
    """
    computes inverse document frequency. IDF is defined as the
    logarithm of the total number of documents in the corpus over the
    number of documents containing the search term:
    log(all documents/documents containing the search term)

    Note that if *no* document contains the search term, it would result
    in a division by zero. This is mitigated by adding 1 to the
    denominator in that case.

    Parameters:
        term: a string containing the search term
        corpus: a list of lists; the outer list is the corpus, while the
            inner lists should represent the document texts, split into
            tokens (make sure that punctuation is split, too!)

    Return Value: a float representing the idf value
    """
    from math import log
    documents_with_term = 0
    for document in corpus:
        for token in document:
            if token == term:
                documents_with_term += 1
                break
    try:
        return log(len(corpus)/documents_with_term)
    except ZeroDivisionError:
        return log(len(corpus) / 1 + documents_with_term)


def tfidf(term, document, corpus):
    """
    computes tf-idf (term frequency-inverse document frequency).
    It can be used to indicate how important the term is in the
    specified document in the specified corpus.

    Parameters:
        term: a string containing the search term
        document: a list representing the document text, split into tokens
            (make sure that punctuation is split, too!)
        corpus: a list of lists; the outer list is the corpus, while the
            inner lists should represent the document texts, split into
            tokens (make sure that punctuation is split, too!)

    Return Value: a float representing the tf-idf value
    """
    return tf(term, document) * idf(term, corpus)
