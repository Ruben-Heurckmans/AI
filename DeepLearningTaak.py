import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data (similar to your previous code)
# ...

# Load and compile the model
def create_model():
    model = models.Sequential()
    # Define your model architecture
    # ...
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# EDA and data preparation
# ...

# Split the data into train, validation, and test sets
# ...

# Train the model
@st.cache(allow_output_mutation=True)
def train_model(epochs):
    model = create_model()
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32)
    return model, history

# Streamlit app
def main():
    st.title("Your Streamlit App")

    # EDA visualization
    st.subheader("Exploratory Data Analysis (EDA)")
    # Visualize EDA components
    # ...

    # Model training controls
    st.sidebar.subheader("Model Training Controls")
    epochs = st.sidebar.slider("Number of Epochs", 1, 50, 10)  # You can adjust the range and default values
    train_button = st.sidebar.button("Train Model")

    if train_button:
        trained_model, training_history = train_model(epochs)

        # Visualize training/validation error
        st.subheader("Training and Validation Error")
        # Plot training and validation error
        plt.figure(figsize=(10, 6))
        plt.plot(training_history.history['loss'], label='Training Loss')
        plt.plot(training_history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        st.pyplot()

        # Evaluate the model on the test set and display the confusion matrix
        st.subheader("Model Evaluation on Test Set")
        test_loss, test_accuracy = trained_model.evaluate(X_test, y_test)
        st.write(f'Test Loss: {test_loss:.4f}')
        st.write(f'Test Accuracy: {test_accuracy:.4f}')

        y_pred_probs = trained_model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        conf_mat = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        st.subheader("Confusion Matrix")
        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot()

if __name__ == "__main__":
    main()
