# 1. Install and Load Dependencies

# Install Dependencies
!pip install torch torchaudio librosa matplotlib scikit-learn

# Load Dependencies
import os
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# 2. Build Data Loading Function

# Define Paths to Files
CAPUCHIN_FILE = os.path.join('data', 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')

# Dataloading Function
def load_wav_16k_mono(filename):
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return torch.tensor(wav)

# 3. Plot Wave

# Plot the Waveforms
wave = load_wav_16k_mono(CAPUCHIN_FILE)
nwave = load_wav_16k_mono(NOT_CAPUCHIN_FILE)
plt.plot(wave.numpy())
plt.plot(nwave.numpy())
plt.show()

# 4. Create Dataset and DataLoader

# Custom Dataset Class
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav = load_wav_16k_mono(self.file_paths[idx])
        label = self.labels[idx]
        return wav, torch.tensor(label)

# Prepare Data
POS = os.path.join('data', 'Parsed_Capuchinbird_Clips')
NEG = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips')

pos_files = [os.path.join(POS, f) for f in os.listdir(POS)]
neg_files = [os.path.join(NEG, f) for f in os.listdir(NEG)]

pos_labels = [1] * len(pos_files)
neg_labels = [0] * len(neg_files)

files = pos_files + neg_files
labels = pos_labels + neg_labels

train_files, test_files, train_labels, test_labels = train_test_split(files, labels, test_size=0.3, random_state=42)

train_dataset = AudioDataset(train_files, train_labels)
test_dataset = AudioDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. Build Preprocessing Function (Optional for Spectrograms)

# Convert to Spectrogram
def preprocess(wav):
    spectrogram = librosa.feature.melspectrogram(y=wav.numpy(), sr=16000, n_mels=128)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return torch.tensor(spectrogram)

# 6. Build and Train the Model

# Define Model
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*128*128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = SimpleCNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train(model, loader):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs.float())
            preds = (outputs.squeeze() > 0.5).float()
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())
    return precision_score(y_true, y_pred), recall_score(y_true, y_pred)

for epoch in range(5):
    train_loss = train(model, train_loader)
    precision, recall = evaluate(model, test_loader)
    print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

# 7. Make Predictions

# Predict on New Data
def predict(model, wav):
    model.eval()
    with torch.no_grad():
        spectrogram = preprocess(wav)
        spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)
        output = model(spectrogram.float())
        return 1 if output > 0.5 else 0

mp3 = 'path_to_mp3_file'
wav = load_wav_16k_mono(mp3)
prediction = predict(model, wav)
print("Prediction:", prediction)

# 8. Export Results

# Save to CSV
import csv

results = {'recording_00.mp3': prediction}

with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in results.items():
        writer.writerow([key, value])
