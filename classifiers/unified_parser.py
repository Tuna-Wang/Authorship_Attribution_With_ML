import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from utils.config import Config

class TextDataset(Dataset):
    def __init__(self, logger, df, max_len=None):
        self.logger = logger
        self.df = df.copy()
        self.max_len = max_len
        self.vocab = {'<pad>': 0, '<unk>': 1}
        self.label_encoder = LabelEncoder()
        
        # Prepare data during initialization
        self._clean_text()
        self._build_vocab()
        self._encode_labels()
    
    def _clean_text(self):
        """Standard text cleaning for both models"""
        self.df['text'] = self.df['text'].apply(
            lambda x: ' '.join(word.lower() for word in str(x).split()))
        self.df['text'] = self.df['text'].str.replace('[^\w\s]', '', regex=True)
        self.df['text'] = self.df['text'].str.replace('\d+', '', regex=True)
        self.df['text'] = self.df['text'].str.replace('\s+', ' ', regex=True).str.strip()
    
    def _build_vocab(self):
        """Build vocabulary that works for both models"""
        all_words = [word for text in self.df['text'] for word in text.split()]
        word_counts = Counter(all_words)
        
        # Include words that appear at least 2 times
        for word, count in word_counts.items():
            if count >= 2 and word not in self.vocab:
                self.vocab[word] = len(self.vocab)
    
    def _encode_labels(self):
        """Standard label encoding for both models"""
        self.df['encoded_label'] = self.label_encoder.fit_transform(self.df['label'])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """For CNN model - returns tensors"""
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['encoded_label']
        
        tokens = text.split()[:self.max_len] if self.max_len else text.split()
        encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
        if self.max_len:
            padded = encoded + [self.vocab['<pad>']] * (self.max_len - len(encoded))
            return torch.tensor(padded, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        return encoded, label
    
    def get_texts_and_labels(self):
        """For N-gram model - returns raw cleaned texts and labels"""
        return self.df['text'].tolist(), self.df['encoded_label'].tolist()
    
    def split_data(self, train_size=0.8):
        """Stratified split that works for both models"""
        train_dfs, test_dfs = [], []
        
        for class_name in self.label_encoder.classes_:
            class_data = self.df[self.df['label'] == class_name]
            train, test = train_test_split(
                class_data,
                test_size=1-train_size,
                random_state=42
            )
            train_dfs.append(train)
            test_dfs.append(test)
        
        train_df = pd.concat(train_dfs).sample(frac=1, random_state=42)
        test_df = pd.concat(test_dfs).sample(frac=1, random_state=42)
        # save the train and test dataset
        train_df.to_csv(os.path.join(Config.output_data_folder(), 'train_dataset.csv'), index=False)
        test_df.to_csv(os.path.join(Config.output_data_folder(), 'test_dataset.csv'), index=False)
        
        return train_df, test_df