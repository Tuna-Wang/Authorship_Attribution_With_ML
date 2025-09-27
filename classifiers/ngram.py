
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import joblib
import os
from utils.config import Config
import re
import csv



from classifiers.unified_parser import TextDataset

class NgramModel():
    def __init__(self, logger, df, ngram_range, max_features, train_dataset, test_dataset):
        self.logger = logger
        self.df = df
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


    def train_ngram_model(self, print_features=False):
        self.logger.info('Training N-gram model...')
        
        # Create TextDataset and get cleaned texts and labels
        train_data = TextDataset(self.logger, self.train_dataset, 10)
        texts, labels = train_data.get_texts_and_labels()
        label_encoder = train_data.label_encoder
        
        # Create Tf-IDF vectorizer
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        train_vectors = vectorizer.fit_transform(texts)

        # Train a Naive Bayes classifier
        model = MultinomialNB()
        model.fit(train_vectors, labels)

        if print_features:
            self._print_features(vectorizer, train_vectors)

        # Save the model and vectorizer
        joblib.dump((vectorizer, model, label_encoder), os.path.join(Config.model_save_path(), 'ngram_model.pkl'))
        self.logger.info('N-gram model trained and saved successfully.')
        return model, vectorizer, label_encoder

    def predict(self):
        self.logger.info('Predicting using N-gram model...')
        # Create TextDataset and get cleaned texts and labels
        test_data = TextDataset(self.logger, self.test_dataset, 10)
        texts, labels = test_data.get_texts_and_labels()
        label_encoder = test_data.label_encoder
        
        # Load the model and vectorizer
        vectorizer, model, label_encoder = joblib.load(os.path.join(Config.model_save_path(), 'ngram_model.pkl'))
        
        # Transform the test data
        test_vectors = vectorizer.transform(texts)
        
        # Make predictions
        predictions = model.predict(test_vectors)
        
        # Calculate accuracy
        #accuracy = accuracy_score(labels, predictions)
        #self.logger.info(f'Accuracy: {accuracy:.4f}')
        #print(f'Accuracy: {accuracy:.4f}')
        # Print classification report
        #report = classification_report(labels, predictions, target_names=label_encoder.classes_)
        #self.logger.info(report)
        
        return predictions, labels
    
    def make_one_predict(self, input_text):
        vectorizer, classifier, label_encoder = joblib.load(os.path.join(Config.model_save_path(), 'ngram_model.pkl'))
        text = self._prepare_raw_text(input_text)
        text_vector = vectorizer.transform([text])
        prediction = classifier.predict(text_vector)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        return predicted_label
    
    def store_result(self):
        res_dict = {}
        for index, row in self.test_dataset.iterrows():
            text = row['text']
            stop = stopwords.words('english')
            text = ' '.join(word for word in text.split() if word not in stop)
            label = row['label']
            parser_text = self._prepare_raw_text(text)
            prediction = self.make_one_predict(parser_text)
            res_dict[index] = {
                'text': text,
                'label': label,
                'prediction': prediction
            }

        output_path = os.path.join(Config.result_save_path(), 'ngram_results.csv')
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'label', 'prediction'])
            writer.writeheader()
            for index, row in res_dict.items():
                writer.writerow({
                    'text': row['text'],
                    'label': row['label'],
                    'prediction': row['prediction']
                })
        
        
        return res_dict

    
    @staticmethod
    def _print_features(vectorizer, train_vectors):
        # Get feature names and their corresponding TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = train_vectors.sum(axis=0).A1
        tfidf_scores_dict = dict(zip(feature_names, tfidf_scores))

        # Sort the features by their TF-IDF scores in descending order
        sorted_tfidf_scores = sorted(tfidf_scores_dict.items(), key=lambda item: item[1], reverse=True)

        # Print the top n highest frequency n-grams
        top_n = 10  
        print(f"Top {top_n} highest frequency n-grams:")
        for ngram, score in sorted_tfidf_scores[:top_n]:
            print(f"{ngram}: {score}")
        return 

    @staticmethod
    def _prepare_raw_text(input_text):
        # Convert to lowercase
        input_text = ' '.join(word.lower() for word in input_text.split())
        # Remove punctuation
        input_text = re.sub(r'[^\w\s]', '', input_text)
        # Remove numbers
        input_text = re.sub(r'\d+', '', input_text)
        # Remove extra spaces
        input_text = re.sub(r'\s+', ' ', input_text)
        return input_text

    

            








