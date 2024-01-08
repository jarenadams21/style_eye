# Data handling modules
import pandas as pd
from pathlib import Path

# ML / Pytorch modules
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# Custom modules
import pre_processing # image augmentation
import cnn # cnn model

# App modules
import gradio as gr



# Navigate to directory holding art
download_path_for_art = Path.cwd() / 'wikiart'

# Paths to the CSV files
train_csv_path = download_path_for_art / 'CSVs' / 'wikiart_csv' / 'style_train.csv'
val_csv_path = download_path_for_art / 'CSVs' / 'wikiart_csv' / 'style_val.csv'

# Load CSV files into pandas DataFrames
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Sample 10% of the data
train_df_sampled = train_df.sample(frac=0.25)
val_df_sampled = val_df.sample(frac=0.25)

# Create dataset objects from the sampled DataFrames
ArtDataset = pre_processing.ArtDataset
# Control augment flags (edit image or not), data frames (dataset you are using), and current directory (grabs correct file path)
train_dataset = ArtDataset(data_frame=train_df_sampled, root_dir=str(download_path_for_art), augmentFlag=True)
val_dataset = ArtDataset(data_frame=val_df_sampled, root_dir=str(download_path_for_art), augmentFlag=False)

# Filter out None values from the datasets
train_dataset = [item for item in train_dataset if item[0] != None]
val_dataset = [item for item in val_dataset if item[0] != None]

# Print details of the first few elements in the train dataset
print('First few samples from the train dataset:')
for i in range(3):  # Adjust the range as needed
    image, label = train_dataset[i]
    print(f"Sample {i}: Image shape: {image.shape}, Label: {label}")

# Print details of the first few elements in the validation dataset
print('First few samples from the validation dataset:')
for i in range(3):  # Adjust the range as needed
    image, label = val_dataset[i]
    print(f"Sample {i}: Image shape: {image.shape}, Label: {label}")

# Create data loaders
train_dl = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=24, shuffle=False)

## TRAINING & INFERENCE
style_model = cnn.Classifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
style_model = style_model.to(device)
# Check that it is on Cuda
next(style_model.parameters()).device

# Counter function to keep track of the best model throughout training, and to terminate if val loss does not improve across 5 epochs
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint_model.pth')
        self.val_loss_min = val_loss

def training(model, train_dl, val_dl, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0

        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            running_corrects += torch.sum(predictions == labels.data)

        epoch_loss = running_loss / len(train_dl.dataset)
        epoch_acc = running_corrects.double() / len(train_dl.dataset)

        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predictions = torch.max(outputs, 1)
                val_corrects += torch.sum(predictions == labels.data)

        val_loss = val_loss / len(val_dl.dataset)
        val_acc = val_corrects.double() / len(val_dl.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        scheduler.step(val_loss)

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Finished Training')

# Usage
num_epochs = 12
training(style_model, train_dl, val_dl, num_epochs)

# ----------------------------
# Inference
# ----------------------------
def inference (model, val_dl):
  correct_prediction = 0
  total_prediction = 0

  # Disable gradient updates
  with torch.no_grad():
    for data in val_dl:
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Get predictions
      outputs = model(inputs)

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

# Run inference on trained model with the validation set
inference(style_model, val_dl)