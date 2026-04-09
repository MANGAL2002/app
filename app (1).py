import streamlit as st
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import seaborn as sns
import pandas as pd
import gdown
import os

st.set_page_config(page_title="EEG DL App", layout="wide")
st.title("🧠 EEG Deep Learning Dashboard")

# ===============================
# 🔗 INPUT: GOOGLE DRIVE FILE ID
# ===============================
file_id = st.text_input("Enter Google Drive File ID")

if st.button("Load & Run"):

    if file_id == "":
        st.error("Please enter File ID")
        st.stop()

    # ===============================
    # 📥 DOWNLOAD FROM DRIVE (FIXED)
    # ===============================
    output = "data.edf"
    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        if not os.path.exists(output):
            with st.spinner("Downloading dataset from Google Drive..."):
                gdown.download(url, output, quiet=False, fuzzy=True)

        st.success("✅ File downloaded successfully!")

    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        st.stop()

    # ===============================
    # 📊 LOAD EEG DATA
    # ===============================
    try:
        raw = mne.io.read_raw_edf(output, preload=True)
        raw.pick_types(eeg=True)
    except:
        st.error("❌ Error reading EDF file. Check file.")
        st.stop()

    st.subheader("📄 EEG Info")
    st.write(raw.info)

    # ===============================
    # ⚙️ PREPROCESSING
    # ===============================
    raw.filter(0.5, 40)
    raw.notch_filter(20)
    raw.resample(100)

    # ===============================
    # ⏱️ CREATE EPOCHS
    # ===============================
    def create_epochs(raw, duration=2):
        events = mne.make_fixed_length_events(raw, duration=duration)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=duration,
                            baseline=None, preload=True)
        return epochs

    epochs = create_epochs(raw)
    data = epochs.get_data()

    # Normalize
    data = (data - np.mean(data)) / np.std(data)

    st.subheader("📐 Data Shape")
    st.write(data.shape)

    # ===============================
    # 📈 EEG PLOT
    # ===============================
    fig1, ax1 = plt.subplots()
    ax1.plot(data[0][0])
    ax1.set_title("EEG Signal Sample")
    st.pyplot(fig1)

    # ===============================
    # 🧠 MODEL
    # ===============================
    labels = np.random.randint(0, 2, len(data))

    class EEG_CNN(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv1d(channels, 16, 3)
            self.pool = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(16, 32, 3)
            self.fc1 = nn.Linear(32 * 24, 64)
            self.fc2 = nn.Linear(64, 2)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    if st.button("Train Model"):

        channels = data.shape[1]
        model = EEG_CNN(channels)

        X = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        loss_list = []
        accuracy_list = []

        st.subheader("🏋️ Training Progress")

        for epoch in range(5):
            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, axis=1)
            acc = (preds == y).float().mean().item()

            loss_list.append(loss.item())
            accuracy_list.append(acc)

            st.write(f"Epoch {epoch+1} → Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

        preds = preds.numpy()
        y_true = y.numpy()

        accuracy = (preds == y_true).mean()
        st.success(f"✅ Final Accuracy: {accuracy:.2f}")

        # ===============================
        # 📊 VISUALS
        # ===============================

        col1, col2 = st.columns(2)

        with col1:
            fig2, ax2 = plt.subplots()
            cm = confusion_matrix(y_true, preds)
            ConfusionMatrixDisplay(cm).plot(ax=ax2)
            ax2.set_title("Confusion Matrix")
            st.pyplot(fig2)

        with col2:
            probs = torch.softmax(outputs, dim=1)[:,1].detach().numpy()
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)

            fig3, ax3 = plt.subplots()
            ax3.plot(fpr, tpr)
            ax3.set_title(f"ROC Curve (AUC={roc_auc:.2f})")
            st.pyplot(fig3)

        # Loss graph
        fig4, ax4 = plt.subplots()
        ax4.plot(loss_list)
        ax4.set_title("Training Loss")
        st.pyplot(fig4)

        # Accuracy graph
        fig5, ax5 = plt.subplots()
        ax5.plot(accuracy_list)
        ax5.set_title("Accuracy")
        st.pyplot(fig5)

        # Multi-channel EEG
        fig6, ax6 = plt.subplots()
        for i in range(min(5, data.shape[1])):
            ax6.plot(data[0][i] + i*5)
        ax6.set_title("Multi-Channel EEG")
        st.pyplot(fig6)

        # Heatmap
        sample_data = data[:200]
        df = pd.DataFrame({
            f"Ch_{i}": sample_data[:, i, :].mean(axis=1)
            for i in range(sample_data.shape[1])
        })
        df["Label"] = labels[:200]

        fig7, ax7 = plt.subplots()
        sns.heatmap(df.corr(), ax=ax7)
        ax7.set_title("Feature Correlation Heatmap")
        st.pyplot(fig7)
