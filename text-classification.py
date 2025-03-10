import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration parameters
MAX_FEATURES = 10000  # Maximum number of words to keep
MAX_LEN = 200  # Maximum length of each text sample
EMBEDDING_DIM = 100  # Embedding dimension
BATCH_SIZE = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.2

# Function to load and preprocess the data
def load_data(data_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(data_path)
    
    # For this example, let's focus on binary classification
    # We'll classify a comment as toxic if it falls under any toxic category
    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['is_toxic'] = df[toxic_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    
    # Clean text
    df['cleaned_text'] = df['comment_text'].apply(clean_text)
    
    return df

def clean_text(text):
    """Clean and preprocess text."""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    else:
        return ""

# Function to prepare the data for training
def prepare_data(texts, labels, max_features, max_len):
    """Tokenize texts and prepare for training."""
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    return padded_sequences, labels, tokenizer

# Build the model
def build_model(max_features, embedding_dim, max_len):
    """Build a text classification model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_features, embedding_dim, input_length=max_len),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to plot training history
def plot_history(history):
    """Plot training & validation accuracy and loss values."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print classification metrics."""
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Toxic', 'Toxic'],
                yticklabels=['Not Toxic', 'Toxic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# Main execution function
def main():
    # Path to the dataset (update this path)
    data_path = 'train.csv'  # This will be in the Kaggle dataset you download
    
    print("Loading data...")
    df = load_data(data_path)
    
    # Check class balance
    class_counts = df['is_toxic'].value_counts()
    print(f"Class distribution:\n{class_counts}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'].values, 
        df['is_toxic'].values, 
        test_size=0.2, 
        random_state=42,
        stratify=df['is_toxic']
    )
    
    print("Preparing data...")
    X_train_seq, y_train, tokenizer = prepare_data(X_train, y_train, MAX_FEATURES, MAX_LEN)
    X_test_seq, y_test, _ = prepare_data(X_test, y_test, MAX_FEATURES, MAX_LEN)
    
    print("Building model...")
    model = build_model(MAX_FEATURES, EMBEDDING_DIM, MAX_LEN)
    model.summary()
    
    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train_seq, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping]
    )
    
    # Plot training history
    plot_history(history)
    
    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(model, X_test_seq, y_test)
    
    # Save the model
    model.save('toxic_comment_classifier.h5')
    print("Model saved as 'toxic_comment_classifier.h5'")
    
    # Save the tokenizer
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer saved as 'tokenizer.pickle'")
    
    # Example prediction function
    def predict_toxicity(text):
        cleaned = clean_text(text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        pred = model.predict(padded)[0][0]
        return pred, "Toxic" if pred > 0.5 else "Not Toxic"
    
    # Test with some examples
    test_texts = [
        "This is a normal comment about the topic.",
        "You are so stupid and ignorant!!"
    ]
    
    for text in test_texts:
        pred_value, pred_class = predict_toxicity(text)
        print(f"Text: {text}")
        print(f"Prediction: {pred_class} (confidence: {pred_value:.4f})\n")

if __name__ == "__main__":
    main()