import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="EEG Dashboard", layout="wide")

# ===============================
# CUSTOM CSS (🔥 UI BOOST)
# ===============================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .card {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    }
    h1, h2, h3 {
        color: #00e5ff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 EEG Deep Learning Pro Dashboard")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Controls")

theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])
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

    st.markdown("### 📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<div class='card'>📦 Samples<br><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'>📌 Features<br><h2>{df.shape[1]}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'>🎯 Classes<br><h2>{df.iloc[:, -1].nunique()}</h2></div>", unsafe_allow_html=True)

    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### 📊 Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ===============================
# TAB 2: VISUALIZATION (🔥 PLOTLY)
# ===============================
with tab2:

    st.subheader("📈 Advanced EEG Visualization")

    col1, col2 = st.columns(2)

    # EEG SIGNAL (Interactive)
    with col1:
        fig = px.line(df.iloc[:300, 0], title="EEG Signal")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # CLASS DISTRIBUTION
    with col2:
        fig = px.histogram(df, x=df.columns[-1], title="Class Distribution")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # CORRELATION HEATMAP
    with col3:
        corr = df.corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # SCATTER MATRIX
    with col4:
        sample_df = df.sample(200)
        fig = px.scatter_matrix(sample_df, color=sample_df.columns[-1])
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

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

        st.success(f"🎯 Accuracy: {acc:.2f}")

        col1, col2 = st.columns(2)

        # LOSS CURVE
        with col1:
            fig = px.line(loss_list, title="Loss Curve")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        # ACCURACY
        with col2:
            fig = px.bar(x=["Accuracy"], y=[acc], title="Accuracy")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        # CONFUSION MATRIX
        with col3:
            cm = confusion_matrix(y_test, preds)
            fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        # ROC CURVE
        with col4:
            probs = torch.softmax(model(X_test), dim=1)[:,1].detach().numpy()
            fpr, tpr, _ = roc_curve(y_test, probs)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
            fig.update_layout(title="ROC Curve", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
