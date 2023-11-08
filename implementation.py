
import pycrfsuite
from sklearn.metrics import classification_report
import os

# Step 1: Data Preparation

def load_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    sentences, labels = [], []
    current_sentence, current_labels = [], []
    for line in data:
        if line == '\n':
            sentences.append(current_sentence)
            labels.append(current_labels)
            current_sentence, current_labels = [], []
        else:
            word, _, _, label = line.strip().split()
            current_sentence.append(word)
            current_labels.append(label)
    return sentences, labels

train_sentences, train_labels = load_data_from_file('train')
valid_sentences, valid_labels = load_data_from_file('valid')
test_sentences, test_labels = load_data_from_file('test')

# Step 2: Feature Extraction

def word2features(sentence, index):
    word = sentence[index]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if index > 0:
        prev_word = sentence[index - 1]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word.istitle()': prev_word.istitle(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence
    if index < len(sentence) - 1:
        next_word = sentence[index + 1]
        features.update({
            'next_word.lower()': next_word.lower(),
            'next_word.istitle()': next_word.istitle(),
        })
    else:
        features['EOS'] = True  # End of sentence
    return features

def sent2features(sentence):
    return [word2features(sentence, i) for i in range(len(sentence))]

def sent2labels(labels):
    return labels

X_train = [sent2features(sentence) for sentence in train_sentences]
y_train = [sent2labels(labels) for labels in train_labels]

X_valid = [sent2features(sentence) for sentence in valid_sentences]
y_valid = [sent2labels(labels) for labels in valid_labels]

X_test = [sent2features(sentence) for sentence in test_sentences]
y_test = [sent2labels(labels) for labels in test_labels]

# Step 3: CRF Model Training

def train_crf(X_train, y_train, output_model_path='ner_crf_model.crfsuite'):
    trainer = pycrfsuite.Trainer()
    for x_seq, y_seq in zip(X_train, y_train):
        trainer.append(x_seq, y_seq)
    trainer.set_params({
        'c1': 1.0,
        'c2': 1e-3,
        'max_iterations': 100,
        'feature.possible_transitions': True
    })
    trainer.train(output_model_path)

train_crf(X_train, y_train)

# Step 4: Model Evaluation

def evaluate_crf(X_test, y_test, model_path='ner_crf_model.crfsuite'):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)
    y_pred = [tagger.tag(x_seq) for x_seq in X_test]
    y_test_flat = [label for labels in y_test for label in labels]
    y_pred_flat = [label for labels in y_pred for label in labels]
    report = classification_report(y_test_flat, y_pred_flat)
    return report

evaluation_report = evaluate_crf(X_test, y_test)
print("Evaluation Report:")
print(evaluation_report)
