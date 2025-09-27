import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import re
import csv
from sklearn.metrics import classification_report
from utils.config import Config

class CNNModel(nn.Module):
    """CNN for speaker identification from text"""
    
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout=0.7):
        """
        Initialize CNN model
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            num_filters: Number of filters per convolution
            filter_sizes: List of filter sizes (e.g., [3,4,5] for trigrams, 4-grams, etc.)
            num_classes: Number of speaker classes
            dropout: Dropout probability
        """
        super(CNNModel, self).__init__()
        
        # Embedding layer converts word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Add dropout to embedding layer
        self.embedding_dropout = nn.Dropout(0.3)
        
        # Reduced number of filters for smaller dataset
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim, 
                out_channels=num_filters // 2,  
                kernel_size=fs
            ) for fs in filter_sizes
        ])
        
        # Add batch normalization for better training stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters // 2) for _ in filter_sizes
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final fully connected layer for classification (adjusted for fewer filters)
        self.fc = nn.Linear((num_filters // 2) * len(filter_sizes), num_classes)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training"""
        nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Embedding layer (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.embedding(x)
        x = self.embedding_dropout(x)  
        
        # Rearrange for convolution (batch_size, embed_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply each convolution and max pooling with batch norm
        pooled_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            # Convolution + Batch Norm + ReLU
            conv_out = conv(x)
            conv_out = bn(conv_out)
            conv_out = torch.relu(conv_out)
            # Max pooling over time dimension
            pooled = torch.max(conv_out, dim=2)[0]
            pooled_outputs.append(pooled)
        
        # Concatenate all pooled features
        x = torch.cat(pooled_outputs, dim=1)
        x = self.dropout(x)
        
        # Final classification layer
        return self.fc(x)
    
    def train_model(self, train_dataset, val_dataset=None, epochs=10, batch_size=32, 
                   learning_rate=0.001, device='cpu', patience=3):
        """
        Complete training procedure with early stopping
        Args:
            train_dataset: SpeakerDataset for training
            val_dataset: SpeakerDataset for validation (optional)
            device: 'cpu' or 'cuda'
            patience: Early stopping patience
        """
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        # Move model to device
        self.to(device)
        
        # Loss function and optimizer with weight decay for regularization
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Early stopping variables
        best_val_acc = 0
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            self.train()  # Set model to training mode
            total_loss = 0
            correct = 0
            total = 0
            
            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device)
                
                # Forward pass
                outputs = self(texts)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Print training stats
            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')
            
            # Validation and early stopping
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader, device)
                print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
                
                # Early stopping logic
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    model_path = os.path.join(Config.model_save_path(), 'cnn_model.pth')
                    torch.save(self.state_dict(), model_path)
                    print(f'New best model saved! Val Acc: {val_acc:.4f}')
                else:
                    patience_counter += 1
                    print(f'No improvement. Patience: {patience_counter}/{patience}')
                    
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                # Save model if no validation
                model_path = os.path.join(Config.model_save_path(), 'cnn_model.pth')
                torch.save(self.state_dict(), model_path)
        
    
    def _evaluate(self, data_loader, device='cpu'):
        """Evaluate model on given dataset"""
        self.eval()  # Set model to evaluation mode
        total_loss = 0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():  # Disable gradient calculation
            for texts, labels in data_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = self(texts)
                
                total_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(data_loader), correct / total
    
    def predict(self, text, vocab, label_encoder, max_len=500, device='cpu'):
        """Predict speaker for raw text"""
        # Clean and tokenize text
        text = ' '.join(word.lower() for word in str(text).split())
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to numerical tokens
        tokens = text.split()[:max_len]
        encoded = [vocab.get(token, vocab['<unk>']) for token in tokens]
        padded = encoded + [vocab['<pad>']] * (max_len - len(encoded))
        
        # Convert to tensor and predict
        input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
        with torch.no_grad():
            output = self(input_tensor)
            _, pred_idx = torch.max(output, 1)
            return label_encoder.inverse_transform([pred_idx.item()])[0]
    
    def predict_and_store_results(self, test_dataset, vocab, label_encoder, device='cpu'):
        """
        Predict on test dataset and store results to CSV
        Args:
            test_dataset: DataFrame with test data
            vocab: Vocabulary dictionary from training
            label_encoder: Label encoder from training
            device: Device to run predictions on
        Returns:
            tuple: (predictions_list, ground_truth_list, results_dict)
        """
        cnn_dict = {}
        predictions = []
        ground_truth = []
        
        for index, row in test_dataset.iterrows():
            input_text = row['text']
            # Preprocess the input text
            input_text = re.sub(r'[^\w\s]', '', input_text)
            input_text = re.sub(r'\d+', '', input_text)
            input_text = re.sub(r'\s+', ' ', input_text).strip()
            
            # Get prediction
            prediction = self.predict(input_text, vocab, label_encoder, device=device)
            
            # Store the result
            cnn_dict[index] = {
                'text': input_text,
                'label': row['label'],
                'prediction': prediction
            }
            
            # Collect predictions and ground truth for evaluation
            predictions.append(prediction)
            ground_truth.append(row['label'])
        
        # Save the results to a CSV file
        output_path = os.path.join(Config.result_save_path(), 'cnn_results.csv')
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'label', 'prediction'])
            writer.writeheader()
            for index, row in cnn_dict.items():
                writer.writerow({
                    'text': row['text'],
                    'label': row['label'],
                    'prediction': row['prediction']
                })
        
        return predictions, ground_truth, cnn_dict
    
    def load_model(self, device='cpu', model_path=None):
        """
        Load a previously saved model
        Args:
            device: Device to load model on ('cpu' or 'cuda')
            model_path: Optional custom path to model file
        """
        if model_path is None:
            model_path = os.path.join(Config.model_save_path(), 'cnn_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.load_state_dict(torch.load(model_path, map_location=device))
        self.to(device)
        self.eval()
        print(f"Model loaded successfully from {model_path}")

    
    @property
    def classes_(self):
        """
        Provide classes_ attribute for compatibility with evaluator
        This should be set after training with the label encoder classes
        """
        if hasattr(self, '_classes'):
            return self._classes
        else:
            # Default fallback - should be set properly during training
            return ['0', '1', '2']  # Placeholder
    
    def set_classes(self, label_encoder):
        """Set the classes from label encoder after training"""
        self._classes = label_encoder.classes_

