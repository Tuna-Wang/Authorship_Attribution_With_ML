import os
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

from utils.config import Config
from transformers import BertTokenizer, BertModel




class LogisticClassifier:
    def __init__(self, logger, df_feature, df_original):
        self.logger = logger
        self.df = df_feature.copy()
        self.logistic_df = df_original.copy()
        self.label_encoder = LabelEncoder()

    def train_model_feature(self):
        """Train a multinomial logistic regression model and return predictions for test set.
        
        Returns:
            tuple: (model, scaler, test_results) - Model, scaler, and DataFrame with test data + predictions
        """
        self.logger.info('Training feature-based Logistic Regression model...')
        # Feature and label selection
        feature_subset = self.df.iloc[:, 2:]  
        label = self.label_encoder.fit_transform(self.df['label'])
        # Print out the label mapping
        self.logger.info(f'Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}')
        
        # Split data while preserving original indices
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            feature_subset, label, self.df.index, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Save the trained model
        model_path = os.path.join(Config.model_save_path(), 'logistic_feature_model.pkl')
        
        joblib.dump(model, model_path)
        self.logger.info(f'Feature-based Logistic Regression model saved to {model_path}')
        
        # Create test results DataFrame
        test_results = pd.DataFrame({
            'text': self.logistic_df.iloc[idx_test]['text'].values,
            'true_label': y_test,
            'predicted_label': model.predict(X_test_scaled),
            'probabilities': list(model.predict_proba(X_test_scaled))
        })
        
        # Add the feature columns
        #feature_columns = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        #test_results = pd.concat([test_results, feature_columns], axis=1)
        
        # Save test results to CSV
        test_results.to_csv(os.path.join(Config.result_save_path(), 'logistic_test_results_F.csv'), index=False)
        
        # Evaluate overall performance
        #accuracy = accuracy_score(y_test, test_results['predicted_label'])
        #self.logger.info(f'Accuracy_F: {accuracy:.4f}')
        #self.logger.info(f'Classification Report_F:\n{classification_report(y_test, test_results["predicted_label"], target_names=self.label_encoder.classes_)}')
        
        return model, test_results
    
    def train_model_bert(self):
        self.logger.info('Training BERT-based Logistic Regression model...')
        # Initialize BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Set device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_model = bert_model.to(device)

        # Convert labels to numerical values
        labels = self.label_encoder.fit_transform(self.df['label'])
        self.logger.info(f'Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}')

        # Get all texts first to preserve order
        all_texts = self.df['text'].tolist()
        
        # Process texts in batches for efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(self.df), batch_size):
            batch_texts = self.df['text'].iloc[i:i+batch_size].tolist()
            
            # Tokenize the batch
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = bert_model(**inputs)
            
            # Use mean pooling of last hidden states
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        text_embeddings = np.array(embeddings)
        
        # Split data while preserving text indices
        indices = np.arange(len(self.df))
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            text_embeddings, labels, indices, test_size=0.2, random_state=42
        )
        
        # Train logistic regression
        lr_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        lr_model.fit(X_train, y_train)

        # Save the trained model
        model_path = os.path.join(Config.model_save_path(), 'logistic_bert_model.pkl')
        joblib.dump(lr_model, model_path)
        self.logger.info(f'BERT-based Logistic Regression model saved to {model_path}')
        
        # Overall Evaluate
        #y_pred = lr_model.predict(X_test)
        #accuracy = accuracy_score(y_test, y_pred)

        #self.logger.info(f'Accuracy_BERT: {accuracy:.4f}')
        #self.logger.info(f'Classification Report_BERT:\n{classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)}')

        # Prepare test results with original text
        test_results = pd.DataFrame({
            'text': [all_texts[i] for i in idx_test],  # Get original texts using saved indices
            'true_label': y_test,
            'predicted_label': lr_model.predict(X_test),
            'probabilities': list(lr_model.predict_proba(X_test))
        })
        
        test_results.to_csv(os.path.join(Config.result_save_path(), 'logistic_test_results_B.csv'), index=False)
        
        return lr_model, test_results