import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Download required NLTK data (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Preprocesses raw text by converting to lowercase, removing URLs, 
    stripping special characters, removing stop words, and lemmatizing.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 3. Remove non-alphabetical characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenization (split by space), Stop-word removal, and Lemmatization
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word) for word in words if word not in stop_words
    ]
    
    # 5. Rejoin text
    return ' '.join(cleaned_words)

def load_and_preprocess_data(file_path):
    """
    Loads dataset, combines title and text (if applicable), and applies text cleaning.
    Assumes a CSV with 'text' and 'label' (1 for Real, 0 for Fake).
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Combine title and text if your dataset separates them
    if 'title' in df.columns and 'text' in df.columns:
         df['full_text'] = df['title'] + " " + df['text']
    else:
         df['full_text'] = df['text']
         
    # Drop empty rows
    df = df.dropna(subset=['full_text', 'label'])
    
    print("Cleaning text data. This may take a moment...")
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    return df

def train_and_evaluate_model(df):
    """
    Builds a TF-IDF and Logistic Regression pipeline, trains it, and evaluates performance.
    """
    X = df['cleaned_text']
    y = df['label']
    
    # Split the dataset 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build a Scikit-learn Pipeline
    # This ensures both the vectorizer and the model are packaged together
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=500))
    ])
    
    print("Training the Logistic Regression model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Fake News", "Real News"]))
    
    return pipeline

def main():
    # Example usage:
    # Replace 'fake_news_dataset.csv' with your actual dataset file (e.g., LIAR or FakeNewsNet)
    data_file = 'fake_news_dataset.csv' 
    
    try:
        df = load_and_preprocess_data(data_file)
        model_pipeline = train_and_evaluate_model(df)
        
        # Save the trained pipeline for the Streamlit app
        model_filename = 'fake_news_pipeline.pkl'
        joblib.dump(model_pipeline, model_filename)
        print(f"Model successfully saved as {model_filename}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {data_file}. Please ensure your dataset is in the same directory.")

if __name__ == "__main__":
    main()