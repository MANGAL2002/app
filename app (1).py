import streamlit as st
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import zipfile
import gdown

st.set_page_config(layout="wide")
st.title("🧠 EEG Full Dataset Deep Learning App")

# ===============================
# INPUT ZIP FILE ID
# ===============================
zip_id = st.text_input("Enter Google Drive ZIP File ID")

if st.button("Run Full Dataset"):

    if zip_id == "":
        st.error("Enter ZIP file ID")
        st.stop()

    zip_path = "dataset.zip"
    data_folder = "dataset"

    # ===============================
    # DOWNLOAD ZIP
    # ===============================
    try:
        if not os.path.exists(zip_path):
            url = f"https://drive.google.com/uc?id={zip_id}"
            with st.spinner("Downloading dataset..."):
                gdown.download(url, zip_path, fuzzy=True)

        st.success("ZIP Downloaded!")

    except:
        st.error("Download failed")
        st.stop()

    # ===============================
    # EXTRACT
    # ===============================
    if not os.path.exists(data_folder):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

    st.success("Dataset Extracted!")

    # ===============================
    # LOAD ALL FILES
    # ===============================
    X_all = []
    y_all = []

    files = os.listdir(data_folder)

    psg_files = [f for f in files if "PSG.edf" in f]

    for psg in psg_files:

        hypno = psg.replace("PSG.edf", "Hypnogram.edf")

        if hypno in files:

            try:
                raw = mne.io.read_raw_edf(os.path.join(data_folder, psg), preload=True)
                annot = mne.read_annotations(os.path.join(data_folder, hypno))

                raw.set_annotations(annot)
                raw.pick_types(eeg=True)

                raw.filter(0.5, 40)
                raw.resample(100)

                events, event_id = mne.events_from_annotations(raw)

                epochs = mne.Epochs(raw, events, event_id,
                                    tmin=0, tmax=2,
                                    baseline=None, preload=True)

                data = epochs.get_data()
                labels = epochs.events[:, -1]

                data = (data - np.mean(data)) / np.std(data)

                X_all.append(data)
                y_all.append(labels)

                st.write(f"Loaded: {psg}")

            except:
                st.warning(f"Skipped: {psg}")

    # ===============================
    # MERGE DATA
    # ===============================
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    st.write("Final Dataset Shape:", X_all.shape)

    # ===============================
    # MODEL (ADVANCED)
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

    X = torch.tensor(X_all, dtype=torch.float32)
    y = torch.tensor(y_all, dtype=torch.long)

    num_classes = len(np.unique(y_all))
    model = EEG_CNN(X_all.shape[1], num_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    loss_list = []

    st.subheader("🏋️ Training")

    for epoch in range(10):
        outputs = model(X)
        loss = loss_fn(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        st.write(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

    preds = torch.argmax(outputs, axis=1).numpy()
    acc = (preds == y_all).mean()

    st.success(f"✅ Final Accuracy: {acc:.2f}")

    # ===============================
    # PLOTS
    # ===============================
    fig1, ax1 = plt.subplots()
    ax1.plot(loss_list)
    ax1.set_title("Loss Curve")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    cm = confusion_matrix(y_all, preds)
    ConfusionMatrixDisplay(cm).plot(ax=ax2)
    st.pyplot(fig2)
