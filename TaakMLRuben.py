import streamlit as st
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar
model_selection = st.sidebar.selectbox("Selecteer een model", ["Random Forest", "Support Vector Machine", "K-Nearest Neighbors"])

# Model training and evaluation
if model_selection == "Random Forest":
    st.subheader("Random Forest Model")

    # Train het model met de trainingsdata
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train.values.ravel())

    # Voorspel de labels voor de testdata
    y_pred = random_forest_model.predict(X_test)

    # Bereken de nauwkeurigheid van het model
    accuracy = accuracy_score(y_test.values.ravel(), y_pred)
    st.write("Nauwkeurigheid van het Random Forest-model:", accuracy)

    # Maak een confusion matrix
    conf_matrix = confusion_matrix(y_test.values.ravel(), y_pred)

    # Plot de confusion matrix met seaborn
    st.pyplot(sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False))

elif model_selection == "Support Vector Machine":
    st.subheader("Support Vector Machine Model")

    # Initialize the Support Vector Machine classifier
    svm_classifier = SVC(random_state=42)

    # Train het model met de trainingsdata
    svm_classifier.fit(X_train, y_train.values.ravel())

    # Voorspel de labels voor de testdata
    y_pred_svm = svm_classifier.predict(X_test)

    # Bereken de nauwkeurigheid van het model
    accuracy_svm = accuracy_score(y_test.values.ravel(), y_pred_svm)
    st.write("Nauwkeurigheid van het SVM-model:", accuracy_svm)

    # Maak een confusion matrix voor SVM
    conf_matrix_svm = confusion_matrix(y_test.values.ravel(), y_pred_svm)

    # Plot de confusion matrix met seaborn
    st.pyplot(sns.heatmap(conf_matrix_svm, annot=True, fmt="d", cmap="Greens", cbar=False))

elif model_selection == "K-Nearest Neighbors":
    st.subheader("K-Nearest Neighbors Model")

    # Maak een K-Nearest Neighbors Classifier aan
    knn_model = KNeighborsClassifier(n_neighbors=3)

    # Train het model met de trainingsdata
    knn_model.fit(X_train, y_train.values.ravel())

    # Voorspel de labels voor de testdata
    y_pred_knn = knn_model.predict(X_test)

    # Bereken de nauwkeurigheid van het model
    accuracy_knn = accuracy_score(y_test.values.ravel(), y_pred_knn)
    st.write("Nauwkeurigheid van het KNN-model:", accuracy_knn)

    # Maak een confusion matrix voor KNN
    conf_matrix_knn = confusion_matrix(y_test.values.ravel(), y_pred_knn)

    # Plot de confusion matrix met seaborn
    st.pyplot(sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Reds", cbar=False))
