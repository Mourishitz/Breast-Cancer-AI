import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# configuring data
entries = pd.read_csv('data/entradas_breast.csv')
outs = pd.read_csv('data/saidas_breast.csv')

training_prevision, test_prevision, training_class, test_class = train_test_split(entries, outs, test_size=0.25)

# creating the neural network
classifier = Sequential()

# entry layer
classifier.add(
    Dense(
        units=16,
        activation='relu',
        kernel_initializer='random_uniform',
        input_dim=30
    )
)

# out layer
classifier.add(
    Dense(
        units= 1,
        activation='sigmoid',
    )
)

classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'binary_accuracy'
    ]
)

classifier.fit(
    training_prevision,
    training_class,
    batch_size=10,
    epochs=100
)

previsions = classifier.predict(test_prevision)
previsions = (previsions > 0.5)

accuracy = accuracy_score(test_class, previsions)
matrix = confusion_matrix(test_class, previsions)
result = classifier.evaluate(test_prevision, test_class)
print(result)