import streamlit as st
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

st.set_page_config(layout="wide")
st.title("🧠 Sleep EEG Classification App")

DATA_PATH = "data"

# ===============================
# LOAD FILES
# ===============================
psg_file = None
hypno_file = None

files = os.listdir(DATA_PATH)

for f in files:
    if "PSG.edf" in f:
        psg_file = f
    if "Hypnogram.edf" in f:
        hypno_file = f

if psg_file is None or hypno_file is None:
    st.error("Dataset files not found!")
    st.stop()

st.success(f"Loaded: {psg_file}")

# ===============================
# LOAD DATA
# ===============================
raw = mne.io.read_raw_edf(os.path.join(DATA_PATH, psg_file), preload=True)
annot = mne.read_annotations(os.path.join(DATA_PATH, hypno_file))

raw.set_annotations(annot)
raw.pick_types(eeg=True)

# ===============================
# PREPROCESSING
# ===============================
raw.filter(0.5, 40)
raw.resample(100)

# ===============================
# CREATE EPOCHS
# ===============================
events, event_id = mne.events_from_annotations(raw)

epochs = mne.Epochs(raw, events, event_id,
                    tmin=0, tmax=2,
                    baseline=None, preload=True)

data = epochs.get_data()
labels = epochs.events[:, -1]

# Normalize
data = (data - np.mean(data)) / np.std(data)

st.write("Dataset Shape:", data.shape)

# ===============================
# PLOT EEG
# ===============================
fig1, ax1 = plt.subplots()
ax1.plot(data[0][0])
ax1.set_title("EEG Signal Sample")
st.pyplot(fig1)

# ===============================
# MODEL
# ===============================
class EEG_CNN(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 32, 3)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 24, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

if st.button("Train Model"):

    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    num_classes = len(np.unique(labels))
    model = EEG_CNN(data.shape[1], num_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    loss_list = []

    for epoch in range(10):
        outputs = model(X)
        loss = loss_fn(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        st.write(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

    preds = torch.argmax(outputs, axis=1).numpy()
    acc = (preds == labels).mean()

    st.success(f"✅ Accuracy: {acc:.2f}")

    # ===============================
    # PLOTS
    # ===============================
    fig2, ax2 = plt.subplots()
    ax2.plot(loss_list)
    ax2.set_title("Training Loss")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm).plot(ax=ax3)
    st.pyplot(fig3)
