import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def load_liar_dataset():
   def load_liar_dataset():
    print("Downloading LIAR dataset from Hugging Face...")
    # Add trust_remote_code=True to bypass the security block
    dataset = load_dataset("liar", trust_remote_code=True)
    df = pd.DataFrame(dataset['train'])
    
    # Map LIAR's 6-way classification to Binary (0 = Fake, 1 = Real)
    # LIAR labels: 0=false, 1=half-true, 2=mostly-true, 3=true, 4=barely-true, 5=pants-fire
    def map_label(label):
        if label in [0, 4, 5]: return 0
        return 1
        
    df['label'] = df['label'].apply(map_label)
    
    # Rename 'statement' column to 'text' so it matches our pipeline
    df = df[['statement', 'label']].rename(columns={'statement': 'text'})
    return df

def load_fakenewsnet_dataset():
    print("Downloading FakeNewsNet datasets directly from GitHub...")
    base_url = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/"
    
    # Load all four CSVs from the repository
    datasets = [
        (pd.read_csv(base_url + "politifact_fake.csv"), 0),
        (pd.read_csv(base_url + "gossipcop_fake.csv"), 0),
        (pd.read_csv(base_url + "politifact_real.csv"), 1),
        (pd.read_csv(base_url + "gossipcop_real.csv"), 1)
    ]
    
    frames = []
    for data, label in datasets:
        data['label'] = label
        # Use the article 'title' as the text feature
        frames.append(data[['title', 'label']].rename(columns={'title': 'text'}))
        
    return pd.concat(frames, ignore_index=True)

def main():
    # 1. Load Data
    df_liar = load_liar_dataset()
    df_fnn = load_fakenewsnet_dataset()
    
    # 2. Merge Datasets
    print("Merging datasets into a single training pipeline...")
    final_df = pd.concat([df_liar, df_fnn], ignore_index=True).dropna()
    
    print(f"Total training samples: {len(final_df)}")
    
    # 3. Build the Machine Learning Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, stop_words='english')),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # 4. Train Model
    print("Training the ML model (this may take a minute or two)...")
    pipeline.fit(final_df['text'], final_df['label'])

    # 5. Save the trained pipeline
    joblib.dump(pipeline, 'fake_news_pipeline.pkl')
    print("Success! 'fake_news_pipeline.pkl' has been updated with real benchmark data.")

if __name__ == "__main__":
    main()