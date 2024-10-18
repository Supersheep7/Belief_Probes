import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils import data
import pickle

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
    
    def __init__(self, input_dim, hidden_dim=256, hidden_dim2=128, hidden_dim3=64, output_dim=1, threshold=0.5):
        super(Neural, self).__init__()

        # Architecture

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

        self.llm = "Default"
        self.layers = [x for x in range(layers)]
        self.data = None

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

    def set_llm(new_llm, self):
        self.llm = new_llm

    def cross_validation(self, X, y, cv=5):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        fold_scores = []

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Reshape y to [n_samples, 1]

        for train_index, val_index in kf.split(X):
            X_train, X_val = X_tensor[train_index], X_tensor[val_index]
            y_train, y_val = y_tensor[train_index], y_tensor[val_index]

            # Create PyTorch DataLoader for mini-batch training
            train_loader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(X_train, y_train), 
                batch_size=self.batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(X_val, y_val), 
                batch_size=self.batch_size, shuffle=False
            )

            # Reset a fresh model for each fold
            self.__init__(input_dim=X.shape[1], llm=self.llm, layers=self.layers)
            criterion = self.criterion
            optimizer = self.optimizer

            # Train the model
            self.train(train_loader, criterion, optimizer)

            # Evaluate the model
            fold_score = self.evaluate(val_loader)
            fold_scores.append(fold_score)
            print(f"Fold Score: {fold_score}")

        # Calculate the mean score across all folds
        mean_score = np.mean(fold_scores)
        print(f"Mean Cross-Validation Score: {mean_score}")
        return fold_scores, mean_score

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

    def train(self, data_loader, criterion, optimizer, epochs=5):

        device = self.device
        self.train()
        
        for epoch in range(epochs):
            print("epoch no", epoch)
            running_loss = 0.0
            for X, label in self.train_loader:

                X = X.to(device)
                label = label.to(device)

                probs = self(X, train=True)
                loss = self.criterion(probs, label)

                self.optimizer.zero_grad()
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

                        preds = self(X_val)

                        true_labels += labels_val.cpu().detach().numpy().tolist()
                        pred_labels += preds.cpu().detach().numpy().tolist()

            accuracy = accuracy_score(true_labels, pred_labels)

            print(f'Epoch [{epoch+1}/{5}], Loss: {running_loss/len(self.train_loader)}')
            print("Accuracy", accuracy)

            if loss.item() < best_score:
                best_score = loss.item()
                self.save(self.llm, self.layers) # Self is the best model
                print("saved model with loss", best_score)

    def evaluate(self, data_loader):
        self.eval()  # Set model to evaluation mode
    
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                preds = self(inputs)
            
                # Collect true labels and predictions
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())

        # Convert lists to numpy arrays and flatten them
        all_labels = np.array(all_labels).flatten()
        all_predictions = np.array(all_predictions).flatten()

        # Generate classification report and confusion matrix
        report = classification_report(all_labels, all_predictions, target_names=['Class 0', 'Class 1'])
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf_matrix)

        return report, conf_matrix

    def test(data, self):
        pass

    def save(llm, layer, self):
        with open(f'neural_{llm}_{layer}.pkl', 'wb') as file:
            pickle.dump(self, file)