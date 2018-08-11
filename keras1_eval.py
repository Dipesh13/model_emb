from keras.models import load_model
from keras1 import X_test,y_test

clf = load_model('keras1.h5')

# for article,gt in zip(X_test,y_test):
#     predictions = clf.predict(article)
#     print(predictions,gt)

print(clf.summary())

# for art in X_test:
#     print(clf.predict(art))