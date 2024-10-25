import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import pickle

'''
Here we have the three probes that we will deploy to test for internal representation of belief
Logreg is a simple logistic regressor
MMP is a mass-mean probes as described in Marks & Tegmark 2023
Neural is a tentative copy of SAPLMA as described in Azaria & Mitchell 2023
'''

random.seed(42)

class LogReg():
    
    def __init__(self):
        self.layers = [x for x in range(layers)]
        self.data = None
        self.gold = None
        self.probe = LogisticRegression()

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
            pickle.dump(self, file)

class Mmp():
    
    def __init__(self, layers):
        self.layers = [x for x in range(layers)]
        self.data = None
        self.gold = None
        self.probe = LogisticRegression()
        self.lda = LDA(n_components=2)

    def cross_validation(self, X, y, cv=5, scoring='accuracy'):             # To test
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.fit(), X, y, cv=kf, scoring=scoring)
        return scores
    
    def fit(data, gold, self):
        data = self.lda.fit_transform(data, gold)
        self.probe.fit(data, gold)

    def predict(data, self):
        data = self.lda.transform(data)
        self.probe.predict(data)

    def save(llm, layer, self):
        with open(f'mmp_{llm}_{layer}.pkl', 'wb') as file:
            pickle.dump(self.probe, file)

class Neural(nn.Module):
    
    def __init__(self, train_data, val_data, test_data, input_dim, hidden_dim=256, hidden_dim2=128, hidden_dim3=64, output_dim=1, threshold=0.5):
        super(Neural, self).__init__()

        # Architecture
        self.train_data = train_data            # 70%
        self.val_data = val_data                # 15%
        self.test_data = test_data              # 15%
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.criterion = nn.BCELoss()

        # Hyperparameters

        self._initialize_weights()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = Adam()
        self.threshold = threshold
        self.best = None

        # Data
        self.input_dim = input_dim
        self.llm = "Default"

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    # The following two functions will be useful to pickle the probes with meaningful filenames

    def set_layers(new_layers, self):
        self.layers = new_layers

    def set_llm(new_llm, self):
        self.llm = new_llm

    def cross_validation(self, cv=5):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        fold_scores = []
        data = self.train_data

        for train_index, val_index in kf.split(data):
            
            data_train, data_val = data[train_index], data[val_index]
            
            # Reset a fresh model for each fold
            self.__init__(self.train_data, self.val_data, self.test_data, input_dim=self.input_dim)

            criterion = self.criterion
            optimizer = self.optimizer
            fold_score = self.train(data_train, data_val, criterion, optimizer)
            fold_scores.append(fold_score)
            print(f"Fold Score: {fold_score}")

        mean_score = np.mean(fold_scores)
        print(f"Mean Cross-Validation Score: {mean_score}")
        return fold_scores, mean_score

    def forward(self, data, train=False):
        
        stream = nn.Flatten()(data) # Data should be passed through dataloader 
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
            return probs    # We want to calculate loss 
        else:
            return preds

    def train(self, data_train=None, data_val=None, criterion=nn.BCELoss(), optimizer=Adam(), epochs=5, cross=False):

        device = self.device
        criterion = self.criterion
        optimizer = self.optimizer
        best_score = float('inf')

        # Workaround for self in default argument 
        if data_train is None: 
            data_train = self.train_data
        if data_val is None: 
            data_val = self.val_data

        X_train = data_train[:, 0]
        y_train = data_train[:, 1]
        X_val = data_val[:, 0]
        y_val = data_val[:, 1]

        # Push tensors in dataloader

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)  
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        for epoch in range(epochs):
            
            self.train()
            print("epoch no", epoch)
            running_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                
                outputs = self(X_batch, train=True)  
                loss = criterion(outputs, y_batch)  
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()  

            # Epoch done. Switch to evaluation
                
            self.eval()

            with torch.no_grad():

                true_labels = []
                pred_labels = []

                for X_val, labels_val in val_loader:

                    X_val = X_val.to(device)
                    labels_val = labels_val.to(device)

                    preds = self(X_val)

                    true_labels += labels_val.cpu().detach().numpy().tolist()
                    pred_labels += preds.cpu().detach().numpy().tolist()

                accuracy = accuracy_score(true_labels, pred_labels)
                report = classification_report(true_labels, pred_labels, target_names=['Class 0', 'Class 1'])
                conf_matrix = confusion_matrix(true_labels, pred_labels)

                print(f'Epoch [{epoch+1}/{5}], Loss: {running_loss/len(train_loader)}')
                print("Accuracy", accuracy)
                print("Classification Report:\n", report)
                print("Confusion Matrix:\n", conf_matrix)

                if loss.item() < best_score:
                    best_score = loss.item()
                    best_accuracy = accuracy
                    self.save(self.llm, self.layers)     # Save best model with the name of the llm and the interested layer(s)
                    print("saved model with loss", best_score)

        if cross: 
            return best_accuracy

    def test(self, data_test=None):

        device = self.device

        if data_test is None:
            data_test = self.test_data

        X = data_test[:, 0]
        y = data_test[:, 1]
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)  
        test_dataset = TensorDataset(X_tensor, y_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        self.eval()

        with torch.no_grad():

            true_labels = []
            pred_labels = []

            for X, labels in test_loader:

                X = X.to(device)
                labels = labels.to(device)

                preds = self(X)

                true_labels += labels.cpu().detach().numpy().tolist()
                pred_labels += preds.cpu().detach().numpy().tolist()

            accuracy = accuracy_score(true_labels, pred_labels)
            report = classification_report(true_labels, pred_labels, target_names=['Class 0', 'Class 1'])
            conf_matrix = confusion_matrix(true_labels, pred_labels)

            print("Accuracy", accuracy)
            print("Classification Report:\n", report)
            print("Confusion Matrix:\n", conf_matrix)

        return accuracy, report, conf_matrix

    def save(self, llm, layer):
        with open(f'neural_{llm}_{layer}.pkl', 'wb') as file:
            pickle.dump(self, file)