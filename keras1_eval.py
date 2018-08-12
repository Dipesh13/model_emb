from keras.models import load_model
from keras1 import X_test,y_test
import numpy as np

# clf = load_model('keras1.h5')
clf = load_model('keras2.h5')

# print(clf.summary())

print(clf.predict(np.array(X_test)))