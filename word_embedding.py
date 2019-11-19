from scribe.transcripts.transcript import Transcript
import nltk
import re
from nltk.corpus import stopwords
import os
from nltk.stem import WordNetLemmatizer
import gensim
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from gensim.models.doc2vec import TaggedDocument


# check if we can use multiple worker threads
gensim.models.word2vec.FAST_VERSION

remove_phrases = ['good morning', 'good afternoon', 'good evening', 'good question', 'okay', 'yes', 'thank you']


#################################################################
# Data Cleaning & Model Training
#################################################################


def get_most_recent_version(dirname):
    files = os.listdir(dirname)
    event_id = []
    transcript_id = []
    for file in files:
        idx = [match.start() for match in re.finditer('_', file)]
        event_id.append(int(file[idx[-5] + 1:idx[-4]]))
        transcript_id.append(int(file[idx[-4] + 1:idx[-3]]))
    df = pd.DataFrame(list(zip(event_id, transcript_id, files)),
                      columns=['event_id', 'transcript_id', 'files'])
    recent = df.groupby(['event_id'])['transcript_id'].transform(max) == df['transcript_id']
    filtered_files = df.loc[recent, 'files'].tolist()
    return filtered_files


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocessing(text, training=True):
    """
    Step 1: lowercase
    Step 2: de-contracted in case there is 'll, 's left
    Step 3: remove self-defined phrases which are meaningless
    Step 4: tokenize the input (a long string which can be a document / paragraph / sentence) into a list of words
    Step 5: remove stop words
    Step 6: remove all  Non-Alphanumeric Characters & numbers
    Step 7: lemmatizer
    """
    text = text.lower()
    text = decontracted(text)
    stop_words = set(stopwords.words('english') + ['question', 'questions', 'indiscernible'])
    for word in remove_phrases:
        text = text.replace(word, '')
    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in nltk.word_tokenize(text)]
    filtered_tokens = []
    tokens = [i for i in tokens if i not in stop_words]
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [lemmatizer.lemmatize(ft) for ft in filtered_tokens]
    # Ensure each sentence has at least three words when training. No need for testing question similarity
    if training:
        return stems if len(stems) >= 3 else None
    else:
        return stems


# Train word2vec by year
class TranscriptSentences(object):
    def __init__(self, dirname: str = None, year: str = None):
        self.dirname = dirname
        self.year = year

    def __iter__(self):
        folders = [folder for folder in os.listdir(self.dirname) if folder.isdigit()]
        # take 2018 whole year as example
        months = [m for m in folders if self.year in m]
        for month in months:
            for fname in get_most_recent_version(self.dirname+'/'+month):
                try:
                    t = Transcript(document_path=self.dirname+'/'+month+'/'+fname)
                    t.parse_document()
                    full = t.text['full']
                    sents = [sent for sent in nltk.sent_tokenize(full)]
                    sents = [preprocessing(sent) for sent in sents if preprocessing(sent) is not None]
                    for sent in sents:
                        yield sent
                except Exception:
                    pass
                continue


# Train doc2vec using labeled sentence by year
class TaggedTranscriptSent(object):
    def __init__(self, dirname: str = None, year: str = None):
        self.dirname = dirname
        self.year = year

    def __iter__(self):
        folders = [folder for folder in os.listdir(self.dirname) if folder.isdigit()]
        # take 2018 whole year as example
        months = [m for m in folders if self.year in m]
        for month in months:
            for fname in get_most_recent_version(self.dirname+'/'+month):
                try:
                    t = Transcript(document_path=self.dirname+'/'+month+'/'+fname)
                    t.parse_document()
                    event_id = t.event['Event_ID']
                    full = t.text['full']
                    sents = [sent for sent in nltk.sent_tokenize(full)]
                    sents = [preprocessing(sent) for sent in sents if preprocessing(sent) is not None]
                    for uid, sent in enumerate(sents):
                        yield TaggedDocument(words=sent, tags=[event_id, str(uid)])
                except Exception:
                    pass
                continue


# Train doc2vec using labeled document by year
class TaggedTranscriptDoc(object):
    def __init__(self, dirname: str = None, year: str = None):
        self.dirname = dirname
        self.year = year

    def __iter__(self):
        folders = [folder for folder in os.listdir(self.dirname) if folder.isdigit()]
        # take 2018 whole year as example
        months = [m for m in folders if self.year in m]
        for month in months:
            for fname in get_most_recent_version(self.dirname + '/' + month):
                try:
                    t = Transcript(document_path=self.dirname+'/'+month+'/'+fname)
                    t.parse_document()
                    event_id = t.event['Event_ID']
                    full = t.text['full']
                    yield TaggedDocument(words=preprocessing(full), tags=[event_id])
                except Exception:
                    pass
                continue


def word2vec_training():
    sentences = TranscriptSentences('../resources/data/raw/transcripts', '2018')
    model = gensim.models.Word2Vec(sentences, size=100, window=5, workers=10)
    model.save("../word2vec_2018.model")


def doc2vec_training(style):
    """
    :param style: use a set of tagged sentences as input or tagged document
    :return:
    """
    if style == 'sent':
        it = TaggedTranscriptSent('../resources/data/raw/transcripts', '2018')
        model = gensim.models.Doc2Vec(it, vector_size=100, window=5, min_count=5, workers=10)
        model.save("../doc2vec_2018.model")

    if style == 'doc':
        it = TaggedTranscriptDoc('../resources/data/raw/transcripts', '2018')
        model = gensim.models.Doc2Vec(it, vector_size=100, window=5, min_count=5, workers=10)
        model.save("../doc2vec_2018_doc.model")


#################################################################
# Application: Sentence Similarity
#################################################################


def filter_docs(questions, condition_on_doc):
    # Treat each question (multiple sentences) as a doc, each doc is a bag of words
    cleaned_q = [preprocessing(q, False) for q in questions]
    corpus = [q for q in cleaned_q if q is not None]
    number_of_docs = len(questions)
    if questions is not None:
        corpus = [doc for doc in corpus if condition_on_doc(doc)]
        print("{} docs removed".format(number_of_docs - len(corpus)))
        return corpus
    else:
        print("No questions in transcript")


def document_vector(word2vec_model, doc):
    # Average vector for all words in every document
    # Remove words not in the model vocabulary
    doc = [word for word in doc if word in word2vec_model.wv.vocab]
    return np.mean(word2vec_model[doc], axis=0)


def has_vector_representation(model, doc):
    # check if at least one word of the document is in the word2vec dictionary
    return not all(word not in model.wv.vocab for word in doc)


t = Transcript('../Transcript_SpellcheckedCopy_20190725132146.xml')
t.parse_document()
questions = t.text['question']


def word2vec_similarity(questions):
    word2vec_model = gensim.models.Word2Vec.load("C:/Users/Xdong/Desktop/NLP/word2vec_2018.model")
    word2vec_model.init_sims(replace=True)
    corpus = filter_docs(questions, lambda doc: has_vector_representation(word2vec_model, doc))
    word2vec_similarity = cosine_similarity(np.array([document_vector(word2vec_model, doc) for doc in corpus]))
    np.fill_diagonal(word2vec_similarity, 0)
    result = np.where(word2vec_similarity == np.amax(word2vec_similarity))
    for num in result[0]:
        print(questions[num])


def doc2vec_similarity(questions):
    doc2vec_model = gensim.models.Doc2Vec.load("C:/Users/Xdong/Desktop/NLP/doc2vec_2018_doc.model")
    corpus = filter_docs(questions, lambda doc: has_vector_representation(doc2vec_model, doc))
    doc2vec_similarity = cosine_similarity(np.array([doc2vec_model.infer_vector(doc) for doc in corpus]))
    np.fill_diagonal(doc2vec_similarity, 0)
    result = np.where(doc2vec_similarity == np.amax(doc2vec_similarity))
    for num in result[0]:
        print(questions[num])










