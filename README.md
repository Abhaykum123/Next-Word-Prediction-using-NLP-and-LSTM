# 🧠 Next Word Prediction using LSTM and NLP
This project demonstrates a Next Word Prediction model built using an LSTM (Long Short-Term Memory) network — a type of Recurrent Neural Network (RNN) — to predict the next word in a sequence. It leverages the classic novel The Blue Castle by L. M. Montgomery as the training corpus. The goal is to learn syntactic and semantic patterns in text to generate grammatically plausible continuations.

--- 

### 📌 Project Highlights
<li>
  ✅ Clean and tokenize a novel-length English corpus.
</li>
<li>✅ Convert text into numerical sequences suitable for deep learning.
</li>
<li>✅ Train an LSTM neural network to predict the next word in a sentence.
</li>
<li>✅ Predict and sample text using the trained model.
</li>

---
## 🗂️ Project Files
<li>next_word.ipynb: Main Jupyter notebook with code, training steps, and explanation.
</li>
<li>blue_castle.txt: Raw novel used as the training dataset.
</li>
<li>(Optional: Add requirements.txt, model.h5, etc., as needed)
</li>

---

## 🛠️ Technologies Used
<ul>
<li>Language: Python 3
</li>
  <li>Libraries:
  </li>
  <ul>
    <li>TensorFlow and Keras for deep learning
</li>
<li>NumPy, Pandas for data handling
</li>
  <LI>NLTK or re for text preprocessing
</LI>
  <li>matplotlib for visualization
</li>
  </ul>
</ul>

---
## 📚 Dataset
**Text Source**: Project Gutenberg - [*The Blue Castle*](https://www.gutenberg.org/ebooks/67979)


License: Public Domain

Description: The full English novel is tokenized into sequences for next-word prediction training.

---
## 🧠 Model Architecture
<li>Embedding Layer: Converts words into dense vectors of fixed size.
</li>
<li>LSTM Layer: Captures temporal dependencies between words in a sequence.
</li>
<li>Dense Output Layer: Softmax activation to output probability distribution over vocabulary.
</li>

```
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length),
    LSTM(units=128, return_sequences=False),
    Dense(vocab_size, activation='softmax')
])
```
---
## 🧪 How It Works
<ol>
  <li>
  Preprocessing:  
  </li>
<ul>
  <li>Load the raw .txt file.
</li>
  <li>Lowercase, clean, and tokenize the text.
</li>
  <li>
    Create input sequences of n words to predict the n+1 word.

  </li>
</ul>
<li>
  Create input sequences of n words to predict the n+1 word.
  <ul>
    <li>Use categorical cross-entropy loss and Adam optimizer.</li>
    <li>Train over multiple epochs until convergence.</li>
  </ul>
</li>
<li>Prediction:
<ul>
  <li>Feed a seed text (e.g., “she was very”) to generate the next probable word.
  </li>
  <li>Add temperature sampling to control randomness in predictions.
  </li>
</ul>
</li>
</ol>

---
## 🚀 Getting Started
1. Clone the Repository
```
git clone[ https://github.com/yourusername/next-word-lstm.git
cd next-word-lstm](https://github.com/Abhaykum123/Next-Word-Prediction-using-NLP-and-LSTM)
```
2. Install Dependencies
```
pip install -r requirements.txt
```
If you don’t have requirements.txt, the main libraries are:
```
pip install tensorflow numpy pandas
```
3. Run the Notebook
```
jupyter notebook next_word.ipynb
```

---
## 🎯 Sample Output
```
Input: 'she was very'
Predicted next word: 'happy'

Input: 'i do not'
Predicted next word: 'know'
```
You can extend predictions to full sentences using a looped prediction technique.

---
## 🔄 Future Enhancements
<li>
 Use Bidirectional LSTM for deeper context understanding.
</li>
<li> Implement Beam Search or Top-k Sampling for better sentence generation.
</li>
<li> Train on multiple books or a larger, more diverse corpus.
</li>
<li> Convert it into a Flask or FastAPI web app for interactive demos.
</li>

---
🙋‍♂️ Author
Abhay Kumar
<li> 🔗[LinkedIn](https://www.linkedin.com/in/abhay-kumar-aa1a9129a/)
</li>
<li>🛠️ Learning AI, ML, and Deep Learning from scratch
</li>
<li>📬 Feel free to contribute or raise an issue!
</li>
----
