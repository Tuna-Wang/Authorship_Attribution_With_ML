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
→ Model structure.  
The CNN model consists of 3 parallel branches with kernel sizes [3,4,5], each branch follows Conv → BatchNorm → ReLU → MaxPool pipeline to capture different n-gram patterns (trigrams, 4-grams, 5-grams).

→ Training hyperparameters:   
epochs - 10. 
batch size - 32  
optimizer - ADAM  
base learning rate - 0.001  
early stopping patience - 3

→ Regularization techniques:   
dropout (embedding: 0.3, main: 0.7)  
weight decay - 1e-4  
batch normalization  
early stopping (patience=3)

→ Results
|      | precision | recall | F1 Score |
| :---   | :--- | :---: | ---: |
|Bernard | 0.48 | 0.32 | 0.38 |
|Humprey | 0.42 | 0.24 | 0.31 |
|Jim | 0.60 | 0.82 | 0.69 |
Overall: 0.56.   
weighted avg: 0.52. 

### Discussion over the low performance
#### Data Constraints: Size and Imbalance.   
The entire corpus contains only 991 lines. Authorship attribution models, especially complex deep learning architectures like BERT and CNNs, require substantially larger datasets to robustly learn subtle stylistic features. Furthermore, the inclusion of many very short lines (e.g., "Yes," "Goodbye") contributes little to stylistic analysis, effectively reducing the amount of meaningful training data.   
The severe imbalance in line distribution (Jim: 506 vs. Humphrey: 267 vs. Bernard: 217) directly accounts for the disparity in performance. Jim, as the majority class, consistently achieved a salient performance (F1 score 0.70+) and a high recall rate across all models. This indicates a strong bias where the models are highly likely to predict Jim for an unknown line. 

#### Shared Context and Register
As dialogue from a single stage play, all text shares a common context (Whitehall politics) and operates within a high, formal, bureaucratic register. The characters are constantly engaged in discussing a specific, concentrated topic. It becomes challenging for the models to isolate true idiosyncratic stylistic markers from the common, domain-specific language that all three educated, professional characters used. 
