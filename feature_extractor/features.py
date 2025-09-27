import os
import re
import nltk
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import Counter
from utils.config import Config
from nltk.tokenize import word_tokenize


# Download required NLTK data
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

class FeatureGetter:
    def __init__(self, logger, df):
        self.logger = logger
        self.df = df.copy()
        self.df_cleaned = self.clean_df(self.df.copy())
        self.Humphrey_top, self.Jim_top, self.Bernard_top, self.Other_top = self.count_word_freq()
        
    def clean_df(self, df):
        """Clean the text data in the DataFrame (for word-based features)"""
        df_cleaned = df.copy()
        df_cleaned['text'] = df_cleaned['text'].str.lower()
        df_cleaned['text'] = df_cleaned['text'].str.replace('[^\w\s]', '', regex=True)
        df_cleaned['text'] = df_cleaned['text'].str.replace('\d+', '', regex=True)
        df_cleaned['text'] = df_cleaned['text'].str.replace('\s+', ' ', regex=True).str.strip()
        output_path = os.path.join(Config.output_data_folder(), 'cleaned_text.csv')
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        return df_cleaned
        
    def count_word_freq(self):
        """Count word frequencies by author label using cleaned text"""
        Humphrey_freq = Counter()
        Jim_freq = Counter()
        Bernard_freq = Counter()
        Other_freq = Counter()
        
        stop = set(stopwords.words('english'))
        
        for _, row in self.df_cleaned.iterrows():
            text = row['text']
            tokens = word_tokenize(text)
            for word in tokens:
                word = word.lower()
                if word in stop:
                    continue
                if word.isdigit():
                    continue
                if word in string.punctuation:
                    continue
                
                if row['label'] == 'Humphrey':
                    Humphrey_freq[word] += 1
                elif row['label'] == 'Jim':
                    Jim_freq[word] += 1
                elif row['label'] == 'Bernard':
                    Bernard_freq[word] += 1
                else:
                    Other_freq[word] += 1
        
        # Get top 10 words for each author
        Humphrey_top = {w for w, freq in Humphrey_freq.most_common(10)}
        Jim_top = {w for w, freq in Jim_freq.most_common(10)}
        Bernard_top = {w for w, freq in Bernard_freq.most_common(10)}
        Other_top = {w for w, freq in Other_freq.most_common(10)}
        
        self.logger.info(f'Top 10 words for Humphrey: {Humphrey_top}')
        self.logger.info(f'Top 10 words for Jim: {Jim_top}')
        self.logger.info(f'Top 10 words for Bernard: {Bernard_top}')
        self.logger.info(f'Top 10 words for Other: {Other_top}')
        
        return Humphrey_top, Jim_top, Bernard_top, Other_top
    
    def common_features(self):
        """Add common text features to DataFrame"""
        # Word and character counts use cleaned text
        self.df['word_count'] = self.df_cleaned['text'].apply(lambda x: len(word_tokenize(x)))
        self.df['char_count'] = self.df_cleaned['text'].apply(lambda x: len(x))
        
        # Sentence-related features use original text with punctuation
        self.df['sentence_count'] = self.df['text'].apply(lambda x: len(self._divide_sentences(x)))
        self.df['avg_word_length'] = self.df_cleaned['text'].apply(self._average_word_length)
        self.df['avg_sentence_length'] = self.df['text'].apply(self._average_sentence_length)
        
        stop = set(stopwords.words('english'))
        self.df['stopword_count'] = self.df_cleaned['text'].apply(
            lambda x: len([word for word in word_tokenize(x) if word.lower() in stop]))
        
    def semantic_features(self):
        """Add semantic features to DataFrame using cleaned text"""
        self.df['humphrey_frequent_word_count'] = self.df_cleaned['text'].apply(
            lambda x: self._count_frequent_words(x, self.Humphrey_top))
        self.df['jim_frequent_word_count'] = self.df_cleaned['text'].apply(
            lambda x: self._count_frequent_words(x, self.Jim_top))
        self.df['bernard_frequent_word_count'] = self.df_cleaned['text'].apply(
            lambda x: self._count_frequent_words(x, self.Bernard_top))
        self.df['other_frequent_word_count'] = self.df_cleaned['text'].apply(
            lambda x: self._count_frequent_words(x, self.Other_top))
        
    def syntactic_features(self):
        """Add syntactic features to DataFrame"""
        # POS tagging uses original text with punctuation
        self.df['adj_count'] = self.df['text'].apply(
            lambda x: len([word for word, pos in pos_tag(word_tokenize(x)) if pos.startswith('JJ')]))
        self.df['adv_count'] = self.df['text'].apply(
            lambda x: len([word for word, pos in pos_tag(word_tokenize(x)) if pos.startswith('RB')]))
        self.df['conjunction_count'] = self.df['text'].apply(
            lambda x: len([word for word, pos in pos_tag(word_tokenize(x)) if pos.startswith('CC')]))
        
        self.df['adj_per_line'] = self.df.apply(
            lambda row: row['adj_count'] / row['sentence_count'] if row['sentence_count'] > 0 else 0, axis=1)
        self.df['adv_per_line'] = self.df.apply(
            lambda row: row['adv_count'] / row['sentence_count'] if row['sentence_count'] > 0 else 0, axis=1)
        self.df['conjunction_per_line'] = self.df.apply(
            lambda row: row['conjunction_count'] / row['sentence_count'] if row['sentence_count'] > 0 else 0, axis=1)
        
    def modality_features(self):
        """Add modality features to DataFrame using original text"""
        self.df['modality_count'] = self.df['text'].apply(
            lambda x: len([word for word, pos in pos_tag(word_tokenize(x)) if pos.startswith('MD')]))
        
        self.df['modality_per_line'] = self.df.apply(
            lambda row: row['modality_count'] / row['sentence_count'] if row['sentence_count'] > 0 else 0, axis=1)

    def add_store_features(self):
        """Add all features and store the DataFrame"""
        self.common_features()
        self.semantic_features()
        self.syntactic_features()
        self.modality_features()
        
        
        output_path = os.path.join(Config.output_data_folder(), 'features.csv')
        self.df.to_csv(output_path, index=False, encoding='utf-8')
        return self.df
    
    @staticmethod
    def _average_word_length(text):
        """Calculate average word length using cleaned text"""
        words = word_tokenize(text)
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    @staticmethod
    def _average_sentence_length(text):
        """Calculate average sentence length in words using original text"""
        sentences = FeatureGetter._divide_sentences(text)
        words = word_tokenize(text)  
        if not sentences:
            return 0
        return len(words) / len(sentences)
    
    @staticmethod
    def _divide_sentences(text):
        """Split text into sentences using original punctuation"""
        sentences = re.split(r'[.!?]+', text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    @staticmethod
    def _count_frequent_words(text, character_frequent_word_set):
        """Count occurrences of frequent words using cleaned text"""
        tokens = word_tokenize(text.lower())
        return sum(1 for word in tokens if word in character_frequent_word_set)