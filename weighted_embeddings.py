from sklearn.feature_extraction.text import TfidfVectorizer

X = ['the red apple', 'the orange', 'the green apple', 'the lemon', 'the plum']
# vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,3),min_df=3,max_df=100,max_features=None)

pl = TfidfVectorizer()
vec = pl.fit_transform(X)

# idf score
word2tfidf = dict(zip(pl.get_feature_names(), pl.idf_))
for word, score in word2tfidf.items():
    print(word, score)

print(pl.vocabulary_)

# print (vec.toarray())
# print (pl.get_feature_names())
