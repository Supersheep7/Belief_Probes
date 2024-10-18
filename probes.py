import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils import data
import pickle

random.seed(42)

class LogReg():
    
    def __init__(self, layers):
        self.layers = [x for x in range(layers)]
        self.data = None
        self.gold = None
        self.probe = LogisticRegression()

    def set_layers(new_layers, self):
        self.layers = new_layers

    def fit(data, gold, self):
        self.probe.fit(data, gold)

    def cross_validation(self, X, y, cv=5, scoring='accuracy'):             # To test
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.fit(), X, y, cv=kf, scoring=scoring) 
        return scores

    def predict(data, self):
        self.probe.predict(data)

    def save(llm, layer, self):
        with open(f'logreg_{llm}_{layer}.pkl', 'wb') as file:
            pickle.dump(self.probe, file)

class Mmp():
    
    def __init__(self, layers):
        self.layers = [x for x in range(layers)]
        self.data = None
        self.gold = None
        self.probe = LogisticRegression()

    def set_layers(new_layers, self):
        self.layers = new_layers

    def cross_validation(self, X, y, cv=5, scoring='accuracy'):             # To test
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.fit(), X, y, cv=kf, scoring=scoring)
        return scores
    
    def fit(data, gold, self):
        lda = LDA(n_components=2)
        data = lda.fit_transform(data, gold)
        self.probe.fit(data, gold)

    def predict(data, self):
        data = LDA.transform(data)
        self.probe.predict(data)

    def save(llm, layer, self):
        with open(f'mmp_{llm}_{layer}.pkl', 'wb') as file:
            pickle.dump(self.probe, file)

class Neural(nn.Module):
    
    def __init__(self, llm, layers, input_dim, hidden_dim=256, hidden_dim2=128, hidden_dim3=64, output_dim=1, threshold=0.5):
        super(Neural, self).__init__()
        self._initialize_weights()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = llm
        self.layers = [x for x in range(layers)]
        self.data = None
        self.gold = None
        self.threshold = threshold
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.criterion = nn.BCELoss()
        self.best = None
        self.optimizer = Adam()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def set_layers(new_layers, self):
        self.layers = new_layers

    def data_loader(self):
        pass

    def cross_validation(self):
        pass

    def forward(self, loader, train=False):
        
        stream = nn.Flatten()(loader[0, :]) # We have to understand what shape the loader passes 
        out = self.fc1(stream)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        logits = self.fc3(out)
        probs = self.sigmoid(logits)
        preds = (probs >= self.threshold).float()

        if train:
            return probs
        else:
            return preds

    def train(self, epochs=5):

        device = self.device
        
        for epoch in range(epochs):
            print("epoch no", epoch)
            running_loss = 0.0
            for X, label in self.train_loader:

                X = X.to(device)
                label = label.to(device)
                self.optimizer.zero_grad()

                probs = self.forward(X, train=True)
                loss = self.criterion(probs, label)

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()

                self.eval()

                with torch.no_grad():

                    true_labels = []
                    pred_labels = []

                    for X_val, labels_val in self.val_loader:

                        X_val = X_val.to(device)
                        labels_val = labels_val.to(device)

                        preds = self.forward(X_val)

                        true_labels += labels_val.cpu().detach().numpy().tolist()
                        pred_labels += preds.cpu().detach().numpy().tolist()

            accuracy = accuracy_score(true_labels, pred_labels)
            cv_scores.append(accuracy)

            print(f'Epoch [{epoch+1}/{50}], Loss: {running_loss/len(self.train_loader)}')
            print("Accuracy", accuracy)
            print("Mean_accuracy", np.mean(cv_scores))

            if loss.item() < best_score:
                best_score = loss.item()
                self.save(self.llm, self.layers) # Self is the best model
                print("saved model with loss", best_score)

    def test(data, self):
        pass

    def save(llm, layer, self):
        with open(f'neural_{llm}_{layer}.pkl', 'wb') as file:
            pickle.dump(self.probe, file)