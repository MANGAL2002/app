import streamlit as st
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import gdown

st.title("🧠 EEG Sleep Stage Classification (Advanced)")

# ===============================
# INPUT: TWO FILE IDS
# ===============================
psg_id = st.text_input("Enter PSG File ID")
hypno_id = st.text_input("Enter Hypnogram File ID")

if st.button("Run Full Model"):

    if psg_id == "" or hypno_id == "":
        st.error("Enter both File IDs")
        st.stop()

    # ===============================
    # DOWNLOAD FILES
    # ===============================
    psg_file = "psg.edf"
    hypno_file = "hypno.edf"

    try:
        if not os.path.exists(psg_file):
            gdown.download(f"https://drive.google.com/uc?id={psg_id}", psg_file, fuzzy=True)

        if not os.path.exists(hypno_file):
            gdown.download(f"https://drive.google.com/uc?id={hypno_id}", hypno_file, fuzzy=True)

    except:
        st.error("Download failed")
        st.stop()

    st.success("Files Loaded")

    # ===============================
    # LOAD DATA
    # ===============================
    raw = mne.io.read_raw_edf(psg_file, preload=True)
    annot = mne.read_annotations(hypno_file)

    raw.set_annotations(annot)
    raw.pick_types(eeg=True)

    # ===============================
    # PREPROCESSING
    # ===============================
    raw.filter(0.5, 40)
    raw.resample(100)

    # ===============================
    # CREATE EVENTS (LABELS)
    # ===============================
    events, event_id = mne.events_from_annotations(raw)

    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=2,
                        baseline=None, preload=True)

    data = epochs.get_data()
    labels = epochs.events[:, -1]

    # Normalize
    data = (data - np.mean(data)) / np.std(data)

    st.write("Data shape:", data.shape)

    # ===============================
    # MODEL (IMPROVED)
    # ===============================
    class EEG_CNN(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv1d(channels, 32, 3)
            self.conv2 = nn.Conv1d(32, 64, 3)
            self.pool = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(64 * 24, 128)
            self.fc2 = nn.Linear(128, len(np.unique(labels)))

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    model = EEG_CNN(data.shape[1])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    loss_list = []

    # ===============================
    # TRAINING
    # ===============================
    for epoch in range(10):
        outputs = model(X)
        loss = loss_fn(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        st.write(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    preds = torch.argmax(outputs, axis=1).numpy()

    acc = (preds == labels).mean()
    st.success(f"✅ Accuracy: {acc:.2f}")

    # ===============================
    # PLOTS
    # ===============================
    fig1, ax1 = plt.subplots()
    ax1.plot(loss_list)
    ax1.set_title("Training Loss")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm).plot(ax=ax2)
    st.pyplot(fig2)
