import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide")
st.title("🧠 EEG Deep Learning Dashboard (Working Version)")

# ===============================
# LOAD DATA (LOCAL CSV)
# ===============================
df = pd.read_csv("EEG_Eye_State_Classification.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ===============================
# VISUALIZATION
# ===============================
st.subheader("📈 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    df.iloc[:, 0].plot(ax=ax1)
    ax1.set_title("EEG Signal Sample")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(x=df.iloc[:, -1], ax=ax2)
    ax2.set_title("Class Distribution")
    st.pyplot(fig2)

# ===============================
# CREATE SEQUENCES
# ===============================
def create_sequences(data, labels, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels[i+seq_length])
    return np.array(X), np.array(y)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_seq, y_seq = create_sequences(X, y)

st.write("Sequence Shape:", X_seq.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

# Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ===============================
# FIXED CNN MODEL (NO ERROR)
# ===============================
class CNN_Model(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, 3)
        self.pool = nn.MaxPool1d(2)

        self._to_linear = None
        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels, seq)

        x = self.pool(torch.relu(self.conv1(x)))

        # Dynamic size calculation
        if self._to_linear is None:
            self._to_linear = x.shape[1] * x.shape[2]
            self.fc1 = nn.Linear(self._to_linear, 64)
            self.fc2 = nn.Linear(64, 2)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = CNN_Model(input_channels=X_train.shape[2])

# ===============================
# TRAIN MODEL
# ===============================
if st.button("🚀 Train Deep Learning Model"):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    loss_list = []

    st.subheader("🏋️ Training Progress")

    for epoch in range(10):
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        st.write(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Prediction
    preds = torch.argmax(model(X_test), axis=1)
    acc = accuracy_score(y_test, preds)

    st.success(f"✅ Accuracy: {acc:.2f}")

    # ===============================
    # PLOTS
    # ===============================

    # Loss Curve
    fig3, ax3 = plt.subplots()
    ax3.plot(loss_list)
    ax3.set_title("Training Loss Curve")
    st.pyplot(fig3)

    # Confusion Matrix
    fig4, ax4 = plt.subplots()
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax4)
    ax4.set_title("Confusion Matrix")
    st.pyplot(fig4)

    # Feature Correlation Heatmap
    fig5, ax5 = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), ax=ax5)
    ax5.set_title("Feature Correlation Heatmap")
    st.pyplot(fig5)
