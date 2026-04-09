import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EEG Eye State App", layout="wide")

st.title("🧠 EEG Eye State Classification Dashboard")

# ===============================
# 📥 LOAD DATA FROM GITHUB
# ===============================
DATA_URL = "https://github.com/MANGAL2002/app/blob/main/EEG_Eye_State_Classification.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ===============================
# 📐 DATA INFO
# ===============================
st.subheader("📐 Dataset Info")
st.write("Shape:", df.shape)
st.write(df.describe())

# ===============================
# 📊 VISUALIZATION
# ===============================
st.subheader("📈 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    df.iloc[:, 0].hist(ax=ax1)
    ax1.set_title("Feature Distribution")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(x=df.iloc[:, -1], ax=ax2)
    ax2.set_title("Target Distribution")
    st.pyplot(fig2)

# ===============================
# 🧠 MODEL TRAINING
# ===============================
st.subheader("🤖 Model Training")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if st.button("Train Model"):

    model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    st.success(f"✅ Accuracy: {acc:.2f}")

    # ===============================
    # 📊 CONFUSION MATRIX
    # ===============================
    st.subheader("Confusion Matrix")

    fig3, ax3 = plt.subplots()
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax3)
    ax3.set_title("Confusion Matrix")
    st.pyplot(fig3)

    # ===============================
    # 📄 CLASSIFICATION REPORT
    # ===============================
    st.subheader("Classification Report")

    report = classification_report(y_test, preds, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # ===============================
    # 📊 FEATURE IMPORTANCE
    # ===============================
    st.subheader("Feature Importance")

    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig4, ax4 = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax4)
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)
