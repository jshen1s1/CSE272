import math
from collections import defaultdict
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import snowball

### tokenize the documents
doc = []
docNames = []
with open('../ohsumed.88-91','r') as f:
    lines = f.readlines()
    for i,line in enumerate(lines):
        if line.strip() == ".W":
            docNames.append(int(lines[i-9].strip()))
            doc.append(lines[i+1].strip())

doc_token = defaultdict(set) # stores word and its corresponding document ids
docid_word = defaultdict(list) # stores word in each doc for word frequency
# remove pronouns, vowels, prepositions
stwords = set(stopwords.words('english'))
# word stemming
stemmer = snowball.SnowballStemmer('english')

for docid, c in zip(docNames, doc):
    for sentence in sent_tokenize(c):
        for word in word_tokenize(sentence):
            word = word.lower()
            if word not in stwords and word.isalnum():
                word = stemmer.stem(word)
                doc_token[word].add(docid)
                docid_word[docid].append(word)



id_word_freq = defaultdict(list) # stores the frequency of each word in each document
for i in docid_word:
    id_word_freq[i] = Counter(docid_word[i])

#print(doc_token)
#print(id_word_freq)

### parse query file
query_content = []
with open('../query.ohsu.1-63','r') as f2:
    lines = f2.readlines()
    for i,line in enumerate(lines):
        if line.strip() == '<top>':
            query_content.append(lines[i+2].strip()[8:] + ' ' + lines[i+4].strip())


query_inverted_index = defaultdict(dict)
for queryid, c in enumerate(query_content):
    for sentence in sent_tokenize(c):
        for word in word_tokenize(sentence):
            word = word.lower()
            if word not in stwords and word.isalnum():
                word = stemmer.stem(word)                
                if word in doc_token:
                    query_inverted_index[queryid][word] = doc_token.get(word)


#print(query_inverted_index)


### Ranking algorithm
# compuet TF
def weighted_term_frequency(term, docid):
    count = id_word_freq[docid].get(term)
    if count:
        return count / float(len(docid_word[docid]))
    else:
        return 0
# compute IDF
def computeIDF(term):
    N = len(doc)
    val = len(doc_token[term])
    return math.log(N / float(val))

### Boolean:
def Boolean():
    final_scores = defaultdict(list)
    for queryid in query_inverted_index:
        scores = defaultdict(list)
        query_terms = query_inverted_index.get(queryid)
        for word in query_terms:
            for docid in query_terms[word]:
                if scores[docid]:
                    scores[docid] += 1
                else:
                    scores[docid] = 1
        soreted_key = sorted(scores, key=scores.get, reverse=1)
        sorted_score = {}
        for k in soreted_key:
            sorted_score[k] = scores[k]
        final_scores[queryid] = sorted_score

    return final_scores

#print(Boolean())

### TF: 
def TF():
    final_scores = defaultdict(list)

    for queryid in query_inverted_index:
        scores = defaultdict(list)
        query_terms = query_inverted_index.get(queryid)
        for word in query_terms:
            for docid in query_terms[word]:
                if scores[docid]:
                    scores[docid] += weighted_term_frequency(word, docid)
                else:
                    scores[docid] = weighted_term_frequency(word, docid)  
        soreted_key = sorted(scores, key=scores.get, reverse=1)
        sorted_score = {}
        for k in soreted_key:
            sorted_score[k] = scores[k]
        final_scores[queryid] = sorted_score

    return final_scores

#print(TF())

### TF*IDF:
def TF_IDF():
    final_scores = defaultdict(list)

    for queryid in query_inverted_index:
        scores = defaultdict(list)
        query_terms = query_inverted_index.get(queryid)
        for word in query_terms:
            for docid in query_terms[word]:
                if scores[docid]:
                    scores[docid] += weighted_term_frequency(word, docid) * computeIDF(word)
                else:
                    scores[docid] = weighted_term_frequency(word, docid) * computeIDF(word)
        soreted_key = sorted(scores, key=scores.get, reverse=1)
        sorted_score = {}
        for k in soreted_key:
            sorted_score[k] = scores[k]
        final_scores[queryid] = sorted_score

    return final_scores
    
#print(TF_IDF())

### Relevence feedback
def Rocchio_r_f():
    final_scores = defaultdict(list)
    bata = 0.65

    for queryid in query_inverted_index:
        scores = defaultdict(list)
        query_terms = query_inverted_index.get(queryid)
        j = 0
        for word in query_terms:
            for docid in query_terms[word]:
                if scores[docid]:
                    scores[docid] += bata * math.log(weighted_term_frequency(word, docid) + 1) * computeIDF(word)
                else:
                    scores[docid] = bata* math.log(weighted_term_frequency(word, docid) + 1) * computeIDF(word)
                j += 1
                if j == 10:
                    break
            j = 0
        soreted_key = sorted(scores, key=scores.get, reverse=1)
        sorted_score = {}
        for k in soreted_key:
            sorted_score[k] = scores[k]
        final_scores[queryid] = sorted_score

    return final_scores

# function write result to log file
def dump(scores, fname):
    with open(fname, "w") as newfile:
        for queryid in scores:
            rank = 1
            for docid in scores[queryid]:
                queryID = "OHSU"+str(queryid+1)
                score = str(scores[queryid].get(docid))
                newfile.write(queryID + " Q0 " + str(docid) + " " + str(rank) + " " + score + " tfidf-Both\n")
                rank += 1
            rank = 1

boolean_rank = Boolean()
TF_rank = TF()
TF_IDF_rank = TF_IDF()
r_f_rank = Rocchio_r_f()

dump(boolean_rank, "boolean.txt")
dump(TF_rank, "TF.txt")
dump(TF_IDF_rank, "TF_IDF.txt")
dump(r_f_rank, "RF.txt")