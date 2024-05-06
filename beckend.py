from collections import defaultdict
from inverted_index_gcp import *
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *


def read_pkl_buckt(name_bucket, namefile):
    """
    Read pickle file from backet

    :param name_bucket: name bucket that store the pickle file
            namefile: name of the pickle file

    :return instance of the pickle file
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(f'206065989_{name_bucket}')
    blob = bucket.get_blob(f'{namefile}.pkl')
    with blob.open("rb") as f:
        ver = pickle.load(f)
    return ver


def read_posting_list(inverted, bucket_name, w):
    """
    Read posting list of word from bucket store

    :param  inverted: inverted index object
            name_bucket: name bucket that store the inverted index
            w: the requested word

    :return posting list - list of tuple (doc_id,tf)
    """
    with closing(MultiFileReader(bucket_name)) as reader:
        locs = inverted.posting_locs[w]
        try:
            b = reader.read(locs, inverted.df[w] * 6)
        except: return []
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * 6:i * 6 + 4], 'big')
            tf = int.from_bytes(b[i * 6 + 4:(i + 1) * 6], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def tokeniz_clean(text):
    """
    Remove stopword and stem the word

    :param text: str

    :return list of clean tokens
    """
    list_token = []
    english_stopwords = frozenset(stopwords.words('english'))
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    stemmer = PorterStemmer()
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    for token in tokens:
        if token not in english_stopwords:
            list_token.append(stemmer.stem(token))
    return list_token


def search_title_backend(query, inverted, inverted_title):
    """
    binary title on unstem title - count how many times word appear in title

    :param query: un clean str query
           inverted,inverted_title: instance of InvertedIndex

    :return a sorted list - [(doc_id,score)]
    """
    temp = defaultdict(int)
    tokens = tokeniz_clean(query)
    res = []
    for i, token in enumerate(tokens):
        try:
            for doc, w in read_posting_list(inverted, "206065989_title_pure", tokens[i]):
                temp[(doc, inverted_title.title[doc])] += w
        except:
            continue
    x = sorted(dict(temp).keys(), key=lambda x: temp[x], reverse=True)
    return x


def binary_title(query, inverted, inverted2, k=150, oneGram=False):
    """
    binary title - count the number of the word appearances in the document title

    :param query: un clean str query
           inverted: instance of InvertedIndex - title
           inverted2: instance of InvertedIndex - title2: bigram title

    :return a sorted list - [(doc_id,score)]
    """
    temp = defaultdict(int)
    tokens = query
    res = []
    for i, token in enumerate(tokens):
        try:
            if oneGram:
                for doc, w in read_posting_list(inverted, "206065989_title3", tokens[i]):
                    x = re.sub(r'[^\w]', ' ', inverted.title[doc]).split(" ")
                    temp[doc] += w / len(x)
            else:
                if len(tokens) - 1 == i: break
                for doc, w in read_posting_list(inverted2, "206065989_title2", tokens[i] + " " + tokens[i + 1]):
                    x = re.sub(r'[^\w]', ' ', inverted.title[doc]).split(" ")
                    temp[doc] += w / len(x)
        except:
            continue
    for doc in sorted(temp.items(), key=lambda x: x[1], reverse=True)[:k]:
        res.append(doc)
    return [(int(i[0]), float(i[1])) for i in res]


def bm25_score(query, inverted, k):
    """
    binary title - count the number of the word appearances in the document title

    :param query: un clean str query
           inverted: instance of InvertedIndex - title
           inverted2: instance of InvertedIndex - title2: bigram title

    :return a sorted list - [(doc_id,score)]
    """
    score = 0.0
    tokens = query
    temp = defaultdict(float)
    res = []
    for token in tokens:
        try:
            for doc_id, freq in read_posting_list(inverted, "206065989_body", token):
                numerator = inverted.df[token] * freq * (1.5 + 1)
                denominator = freq + 1.5 * (1 - 0.75 + 0.75 * inverted.doc_len[doc_id] / inverted.avg)
                temp[doc_id] += (numerator / denominator)
        except:
            continue

    for doc in sorted(temp.items(), key=lambda x: x[1], reverse=True)[:k]:
        res.append(doc)

    res = np.asarray(res)
    try:
        res[:, 1] = (res[:, 1] - np.min(res[:, 1])) / (np.max(res[:, 1]) - np.min(res[:, 1]))
    except:
        pass
    return [(int(i[0]), float(i[1])) for i in res]


def cosin_similarity(query, inverted, k=20):
    """
    Calculate the cosinsimilarity between the query and the relevant docs

    :param query: un clean str query
           inverted: instance of InvertedIndex - body of the text

    :return a sorted list - [(doc_id,score)]
    """
    res = []
    dict_sim = defaultdict(float)
    tokens = query
    for i, t_query in enumerate(tokens):
        for doc, w in read_posting_list(inverted, "206065989_body", t_query):
            try:
                dict_sim[doc] += (w / inverted.doc_len[doc]) * np.log2(inverted.num_doc / inverted.df[t_query])

            except:
                continue

    for doc in dict_sim:
        try:
            dict_sim[doc] = (dict_sim[doc] / np.sqrt((len(tokens))) * inverted.norm[doc])
        except:
            continue

    for doc in sorted(dict_sim.items(), key=lambda x: x[1], reverse=True)[:k]:
        res.append(doc)
    if res[0][1] == 0: return []
    res = np.asarray(res)
    try:
        res[:, 1] = (res[:, 1] - np.min(res[:, 1])) / (np.max(res[:, 1]) - np.min(res[:, 1]))
    except:
        pass
    return [(int(i[0]), float(i[1])) for i in res]


def search_page(list_doc, page_rank):
    """
     Return the page rank of the requests docs

     :param list_doc: a list of docs
            page_rank: dict of page rank by id doc

     :return a sorted list of the ranks of the docs
     """
    list_res = []
    for doc in list_doc:
        list_res.append(page_rank[doc])
    return list_res


def search_view(list_doc, view):
    """
      Return the page view of the requests docs

      :param list_doc: a list of docs
             view: dict of page view by id doc

      :return a sorted list of the number views of the docs
      """
    list_res = []
    for doc in list_doc:
        list_res.append(view[doc])
    return list_res


def binary_anchor_search(query, inverted_anchor, inverted_title):
    """
    Count the number of times we linked in word advice to a document

    :param query: un clean str query
           inverted_anchor: instance of InvertedIndex
           inverted_title: instance of InvertedIndex

    :return a sorted list - [(doc_id,score)]
    """
    res = defaultdict(int)
    tokens = tokeniz_clean(query)
    for token in tokens:
        for tup in read_posting_list(inverted_anchor, "206065989_anchor", token):
            try:
                res[(tup[0], inverted_title.title[tup[0]])] += tup[1]
            except:
                continue
    x = sorted(dict(res).keys(), key=lambda x: res[x], reverse=True)
    return x


def binary_anchor(query, inverted_anchor,k):
    """
    Count the number of times we linked in word advice to a document

    :param query: un clean str query
           inverted_anchor: instance of InvertedIndex
           k: limit the result

    :return a sorted list - [(doc_id,score)] that normelize the score
    """
    z = defaultdict(float)
    res = []
    tokens = query
    for token in tokens:
        for tup in read_posting_list(inverted_anchor, "206065989_anchor", token):
            try:
                z[tup[0]] += tup[1]
            except:
                continue
    for doc in sorted(z.items(), key=lambda x: x[1], reverse=True)[:k]:
        res.append(doc)
    res = np.asarray(res)
    if res[0][1] == 0: return []
    try:
        res[:, 1] = (res[:, 1] - np.min(res[:, 1])) / (np.max(res[:, 1]) - np.min(res[:, 1]))
    except:
        pass
    return [(int(i[0]), float(i[1])) for i in res]


def merge1(tokens, inverted_title, inverted_title2, inverted_body,inverted_anchor, k, oneGram):
    res = defaultdict(float)
    cos = cosin_similarity(tokens, inverted_body, k)
    bm25 = bm25_score(tokens, inverted_body, k)
    anchor = binary_anchor(tokens,inverted_anchor,k)
    title = binary_title(tokens, inverted_title, inverted_title2, k, oneGram)

    for doc_id, val in cos:
        res[doc_id] += val / 16
    for doc_id, val in bm25:
        res[doc_id] += val / 16
    for doc_id, val in title:
        res[doc_id] += val * 7/ 16
    for doc_id, val in anchor:
        res[doc_id] += val * 7 / 16 
    return (res)


def merge2(tokens, inverted_title, inverted_title2, inverted_body,inverted_anchor, k, oneGram):
    res = defaultdict(float)
    oneG = merge1(tokens, inverted_title, inverted_title2, inverted_body,inverted_anchor, k, oneGram=True)
    twoG = binary_title(tokens, inverted_title, inverted_title2, k, oneGram)
    for doc_id, val in oneG.items():
        res[doc_id] += val * 1 / 6
    for doc_id, val in twoG:
        res[doc_id] += val * 5 / 6
    return (res)


def merge3(tokens, inverted_title, inverted_title2, inverted_body,inverted_anchor, k, oneGram):
    res = defaultdict(float)
    oneG = merge1(tokens, inverted_title, inverted_title2, inverted_body,inverted_anchor, k, oneGram=True)
    twoG = binary_title(tokens, inverted_title, inverted_title2, k, oneGram)
    for doc_id, val in oneG.items():
        res[doc_id] += val *1 / 6
    for doc_id, val in twoG:
        res[doc_id] += val * 5 / 6
    return (res)


def search_body_b(query, inverted_body, inverted_title):

    res = []
    dict_sim = defaultdict(float)
    tokens = tokeniz_clean(query)
    for i, t_query in enumerate(tokens):
        try:
            for doc, w in read_posting_list(inverted_body, "206065989_body", t_query):
                try:
                    dict_sim[(doc, inverted_title.title[doc])] += (w / inverted_body.len_doc[doc]) * np.log2(
                        inverted_body.num_doc / inverted_body.df[t_query])
                except:
                    continue
        except:
            continue
    for doc in dict_sim:
        try:
            dict_sim[doc] = (dict_sim[doc] / np.sqrt((len(tokens))) * inverted_body.norm[doc])
        except:
            continue
    x = sorted(dict(dict_sim).keys(), key=lambda x: dict_sim[x], reverse=True)[:100]
    return x


def search_backend(query, inverted_title, inverted_title2, inverted_body,inverted_anchor):
    tokens = tokeniz_clean(query)
    if len(tokens) == 1:
        res = merge1(tokens, inverted_title, inverted_title2, inverted_body,inverted_anchor, 150, oneGram=True)
    elif len(tokens) == 2:
        res = merge2(tokens, inverted_title, inverted_title2, inverted_body,inverted_anchor, 150, False)
    else:
        res = merge3(tokens, inverted_title, inverted_title2, inverted_body,inverted_anchor, 150, False)
    x = sorted(dict(res).items(), key=lambda y: y[1], reverse=True)[:100]
    return [(doc, inverted_title.title[doc]) for doc, val in x]