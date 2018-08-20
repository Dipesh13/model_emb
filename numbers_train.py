# https://pypi.org/project/inflect/
# https://pypi.org/project/num2words/
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from num2words import num2words
import inflect

p = inflect.engine()
# word = p.number_to_words(1234567)
# print(word)

# print(num2words(42))

data1 = []
data2 = []
data3 = []
data4 = []
for a in range(5):
    # print(num2words(i))
    # print(p.number_to_words(i))
    data1.append(num2words(a))

for a in range(6,10):
    # print(num2words(i))
    # print(p.number_to_words(i))
    data2.append(num2words(a))

for a in range(3,8):
    # print(num2words(i))
    # print(p.number_to_words(i))
    data3.append(num2words(a))

for a in range(2,16,4):
    # print(num2words(i))
    # print(p.number_to_words(i))
    data4.append(num2words(a))

dataset = [data1,data2,data3,data4]
print(dataset)

# train model
model = Word2Vec(dataset, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['two'])
# save model
model.save('num_model.bin')
# load model
new_model = Word2Vec.load('num_model.bin')
print(new_model)

# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()