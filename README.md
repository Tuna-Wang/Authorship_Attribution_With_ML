## Experimenting Authorship Attribution with ML
## Dataset
The dataset used in the experiment was the stage play Yes, Minister (Jay and Lynn, 2010). Description of the stage settings and actor movements, retaining only the dialogue lines and character names were excluded. Additionally, annotations about facial expressions and body language from each line were removed, preserving only the spoken text. Punctuation, numerical digits, and stop words were retained initially, as they are considered to be meaningful features that may contribute to a character’s speaking style.  
For data abundancy and speech style analysis purpose, this experiment only selected the three main character of the play:  
Jim - Prime Minister.   
Humphrey - Cabinet Secretary   
Bernard - Principal Private Secretary to the Prime Minister

Final dataset consists of 991 lines (Humphrey: 267, Bernard: 217, Jim: 506)
## Classfiers and Performance
### 1. Logistic Regression  
####  Manual Feature Selection (Baseline Model)
→ Features selected  
• Common features: word count per sentence, character count per sentence, sentence count per lines, average word length, average sentence length, stop word count...  
• Semantic features: we counted the most frequently used word for each character, and for each line, we counted the occurrence of the frequent word.  
• Syntactic features: adjective count, conjunction count, adverb count, and their per- centage per line.  
• Other: modal word count.

→Results  
|      | precision | recall | F1 Score |
| :---   | :--- | :---: | ---: |
|Bernard | 0.45 | 0.13 | 0.20 |
|Humprey | 0.47 | 0.26 | 0.33 |
|Jim | 0.62 | 0.92 | 0.74 |
Overall: 0.59.   
weighted avg: 0.52. 

#### BERT Model
 BERT embeddings were used to transform the text into contextualized vector representations. It is hypothesized that BERT’s ability to capture semantic meaning and perform word sense disambiguation would yield better results.
→Results  
|      | precision | recall | F1 Score |
| :---   | :--- | :---: | ---: |
|Bernard | 0.41 | 0.47 | 0.44 |
|Humprey | 0.48 | 0.37 | 0.42 |
|Jim | 0.70 | 0.74 | 0.72 |
Overall: 0.59.   
weighted avg: 0.58. 

It is hypothesized that BERT’s ability to capture semantic meaning and perform word sense disambiguation would yield better results. Suprisingly, no overall improvement was observed if only consider plain accuracy. However, for Bernard, F1 Score increased a lot, considering Bernard has the least data, BERT model has an significant improvement for minority class performance.  

### 2. N-gram  
N-gram model is proven to be the least effective. Initial experiments with N-gram ranges of 1-3 yielded an accuracy of 0.57. Wright(2014) used n-gram to identify author of emails using the Enron email corpus, Wright tested 5 different sample size and n-grams ranging from 1 to 6 and found that longer n-grams (4-6) work best for small texts, while shorter ones (2-3) excel with larger texts. Inspired by Wright, different ranges of n-gram are tested, from 1-6. However, accuracy did not vary.

### 3. CNN
→ Model structure
The CNN model consists of 3 parallel branches, each branch is a combination of Conv → BatchNorm → ReLU → MaxPool, different kernel sizes were applied to capture different n-gram patterns. 
→ Training hyperparameters:
epochs - 20
batch size - 16
optimizer - ADAM
base learning rate - 0.001
early stopping patience - 3
→ Regularization techniques:
dropout (embedding: 0.3, main: 0.7)
weight decay - 1e-4
batch normalization
early stopping (patience=3)
→ 

