import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(layout="wide")
sns.set_style("whitegrid")

st.title("🧠 EEG Deep Learning Pro Dashboard")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Controls")
seq_length = st.sidebar.slider("Sequence Length", 5, 20, 10)
epochs = st.sidebar.slider("Epochs", 5, 30, 10)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("EEG_Eye_State_Classification.csv")

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Visualization", "🤖 Model"])

# ===============================
# TAB 1: DATA
# ===============================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", len(df))
    col2.metric("Features", df.shape[1])
    col3.metric("Classes", df.iloc[:, -1].nunique())

    st.subheader("Statistics Table")
    st.dataframe(df.describe(), use_container_width=True)

# ===============================
# TAB 2: VISUALIZATION
# ===============================
with tab2:

    st.subheader("📈 EEG Visual Analysis")

    # ===============================
    # ROW 1 (SIDE-BY-SIDE)
    # ===============================
    col1, col2 = st.columns(2)

    # EEG SIGNAL
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6,3))
        ax1.plot(df.iloc[:300, 0], color="#1f77b4")
        ax1.set_title("EEG Signal")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)

    # CLASS DISTRIBUTION
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6,3))
        sns.countplot(x=df.iloc[:, -1], ax=ax2, palette="viridis")
        ax2.set_title("Class Distribution")
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # ===============================
    # ROW 2 (SIDE-BY-SIDE)
    # ===============================
    col3, col4 = st.columns(2)

    # HEATMAP
    with col3:
        fig3, ax3 = plt.subplots(figsize=(6,3))
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax3)
        ax3.set_title("Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    # PAIRPLOT (SMALL)
    with col4:
        sample_df = df.sample(200)
        fig4 = sns.pairplot(sample_df, hue=sample_df.columns[-1], corner=True)
        st.pyplot(fig4)

# ===============================
# DATA PREP
# ===============================
def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels[i+seq_length])
    return np.array(X), np.array(y)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_seq, y_seq = create_sequences(X, y, seq_length)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ===============================
# MODEL
# ===============================
class CNN_LSTM(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 3)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.permute(0,2,1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

model = CNN_LSTM(input_channels=X_train.shape[2])

# ===============================
# TAB 3: MODEL
# ===============================
with tab3:

    if st.button("🚀 Train Model"):

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        loss_list = []

        for epoch in range(epochs):
            outputs = model(X_train)
            loss = loss_fn(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        preds = torch.argmax(model(X_test), axis=1)
        acc = accuracy_score(y_test, preds)

        st.success(f"Accuracy: {acc:.2f}")

        # ===============================
        # RESULT GRAPHS SIDE-BY-SIDE
        # ===============================
        col5, col6 = st.columns(2)

        # LOSS
        with col5:
            fig5, ax5 = plt.subplots(figsize=(6,3))
            ax5.plot(loss_list)
            ax5.set_title("Loss Curve")
            plt.tight_layout()
            st.pyplot(fig5, use_container_width=True)

        # ACCURACY BAR
        with col6:
            fig6, ax6 = plt.subplots(figsize=(6,3))
            ax6.bar(["Accuracy"], [acc])
            ax6.set_title("Accuracy")
            plt.tight_layout()
            st.pyplot(fig6, use_container_width=True)

        # ===============================
        # NEXT ROW
        # ===============================
        col7, col8 = st.columns(2)

        # CONFUSION MATRIX
        with col7:
            fig7, ax7 = plt.subplots(figsize=(6,3))
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt="d", ax=ax7)
            ax7.set_title("Confusion Matrix")
            plt.tight_layout()
            st.pyplot(fig7, use_container_width=True)

        # ROC CURVE
        with col8:
            probs = torch.softmax(model(X_test), dim=1)[:,1].detach().numpy()
            fpr, tpr, _ = roc_curve(y_test, probs)

            fig8, ax8 = plt.subplots(figsize=(6,3))
            ax8.plot(fpr, tpr)
            ax8.set_title("ROC Curve")
            plt.tight_layout()
            st.pyplot(fig8, use_container_width=True)
