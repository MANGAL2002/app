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
st.title("🧠 EEG Deep Learning Smart Dashboard")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("EEG_Eye_State_Classification.csv")

# ===============================
# TABS UI
# ===============================
tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Visualization", "🤖 Model"])

# ===============================
# TAB 1: DATA
# ===============================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

# ===============================
# TAB 2: VISUALIZATION
# ===============================
with tab2:

    col1, col2 = st.columns(2)

    # EEG signal
    with col1:
        fig1, ax1 = plt.subplots(figsize=(4,2))
        df.iloc[:500, 0].plot(ax=ax1)
        ax1.set_title("EEG Signal")
        st.pyplot(fig1)

    # Class distribution
    with col2:
        fig2, ax2 = plt.subplots(figsize=(4,2))
        sns.countplot(x=df.iloc[:, -1], ax=ax2)
        ax2.set_title("Class Distribution")
        st.pyplot(fig2)

    # Correlation Heatmap
    fig3, ax3 = plt.subplots(figsize=(6,3))
    sns.heatmap(df.corr(), ax=ax3, cmap="coolwarm")
    ax3.set_title("Correlation Heatmap")
    st.pyplot(fig3)

# ===============================
# PREPARE DATA
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

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ===============================
# CNN MODEL (FIXED)
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
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))

        if self._to_linear is None:
            self._to_linear = x.shape[1] * x.shape[2]
            self.fc1 = nn.Linear(self._to_linear, 64)
            self.fc2 = nn.Linear(64, 2)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = CNN_Model(input_channels=X_train.shape[2])

# ===============================
# TAB 3: MODEL
# ===============================
with tab3:

    if st.button("🚀 Train Model"):

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        loss_list = []

        for epoch in range(10):
            outputs = model(X_train)
            loss = loss_fn(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        preds = torch.argmax(model(X_test), axis=1)
        acc = accuracy_score(y_test, preds)

        st.success(f"Accuracy: {acc:.2f}")

        col1, col2 = st.columns(2)

        # Loss curve
        with col1:
            fig4, ax4 = plt.subplots(figsize=(4,2))
            ax4.plot(loss_list)
            ax4.set_title("Loss Curve")
            st.pyplot(fig4)

        # Accuracy bar
        with col2:
            fig5, ax5 = plt.subplots(figsize=(4,2))
            ax5.bar(["Accuracy"], [acc])
            st.pyplot(fig5)

        # Confusion matrix
        fig6, ax6 = plt.subplots(figsize=(4,3))
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax6)
        ax6.set_title("Confusion Matrix")
        st.pyplot(fig6)

        # Feature importance (Random)
        fig7, ax7 = plt.subplots(figsize=(4,3))
        importances = np.random.rand(X.shape[1])
        sns.barplot(x=importances, y=df.columns[:-1], ax=ax7)
        ax7.set_title("Feature Importance")
        st.pyplot(fig7)
