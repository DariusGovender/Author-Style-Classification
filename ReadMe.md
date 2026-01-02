# Author-Style-Classification (LSTM NLP Project)

The goal of this project is to develop a Natural Language Processing (NLP) model capable of predicting the **author of a given text snippet based purely on writing style**. The project applies deep learning techniques, specifically a **Long Short-Term Memory (LSTM)** network, to capture long-term textual patterns that distinguish authors.

This project demonstrates how recurrent neural networks can be used for stylistic text classification and forms the foundation of an interactive author prediction bot.

---

## Dataset
The dataset consists of **25,000+ text samples** labelled by author. After data cleaning and balancing, the analysis focuses on two authors:

- **J.K. Rowling**
- **Stephen King**

Each record contains:
- Text content
- Author label

The dataset was evaluated and justified as suitable for LSTM-based modelling due to its sequential structure, sufficient sample size, and stylistic consistency.

---

## Workflow

### 1. Data Cleaning & Preparation
- Removed duplicate records to reduce bias.
- Normalised inconsistent author name spellings.
- Addressed class imbalance by removing underrepresented authors.
- Encoded target labels using `StringIndexer`.

### 2. Text Preprocessing
- Converted Spark DataFrame to Pandas.
- Removed noise (HTML tags, URLs, special characters).
- Expanded contractions and converted text to lowercase.
- Applied lemmatisation to standardise word forms.

### 3. Exploratory Data Analysis (EDA)
- **Univariate analysis**:
  - Class distribution
  - Text length distribution
  - Word frequency analysis using word clouds
- **Bivariate analysis**:
  - Comparison of paragraph lengths between authors

Key insights showed that Stephen King’s writing tends to be longer and more descriptive, while J.K. Rowling’s text is more dialogue-driven.

---

### 4. Feature Engineering & Model Preparation
- Tokenised text data.
- Converted text into padded sequences for uniform input length.
- Split data into training and testing sets (80/20) to prevent data leakage.

---

### 5. Model Training
- Built an LSTM-based neural network including:
  - Embedding layer
  - SpatialDropout1D for regularisation
  - LSTM layer
  - Dense output layer with Softmax activation
- Hyperparameter tuning performed using **KerasTuner (Random Search)**.
- Applied Early Stopping to minimise overfitting.

---

### 6. Evaluation
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix
  - ROC-AUC
- Final model performance:
  - **85% accuracy**
  - **97% recall for Stephen King**
  - **73% recall for J.K. Rowling**
  - **ROC-AUC score: 0.92**

---

## 📈 Results
- The model demonstrates strong discriminative ability between author writing styles.
- Performs exceptionally well at identifying Stephen King’s writing.
- Slight classification bias observed toward Stephen King.
- Minor overfitting present but within acceptable limits.

---

## 🛠️ Technologies Used
- Python  
- PySpark  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- TensorFlow / Keras  
- KerasTuner  

---

## ✅ Conclusion
This project successfully demonstrates how LSTM-based deep learning models can be applied to author classification using textual data. While minor class bias exists, the model achieves strong overall performance and effectively captures stylistic differences between authors.

Future improvements may include:
- Applying class weighting during training
- Expanding the dataset to include more authors
- Experimenting with Transformer-based architectures
- Deploying the model as a web-based application
