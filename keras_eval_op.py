import numpy as np
from keras.models import load_model

X_test = [(12,37)]
X_test1 = [(172,45)]
X_test2 = [(231,57)]
X_test3 = [(463,464)]
X_test4 = [(4,6)]

clf = load_model('keras_operators.h5')

print(clf.predict(np.array(X_test)))
print(clf.predict(np.array(X_test1)))
print(clf.predict(np.array(X_test2)))
print(clf.predict(np.array(X_test3)))
print(clf.predict(np.array(X_test4)))