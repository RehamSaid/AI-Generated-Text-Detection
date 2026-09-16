# AI-Generated Text Detection

A Data Mining course project that classifies a piece of text as **Human-written** or **AI-generated** (GPT-3.5), using classical machine learning techniques.

> Given a piece of text, predict whether a human or an AI wrote it — formulated as a binary text classification problem.

---

## 📌 Problem Statement

With the widespread adoption of large language models such as ChatGPT, distinguishing between human-written and AI-generated text has become an important, practical problem. Applications include:

- **Academic integrity** — detecting AI-assisted assignments and essays
- **Content moderation** — flagging AI-generated misinformation online
- **Publishing & journalism** — verifying authorship and content authenticity

This project builds a full data mining pipeline — from raw text to a trained, evaluated classifier — to address this problem.

---

## 📊 Dataset

**Source:** [Human-AI-Generated Text Corpus](https://github.com/LorenzM97/human-AI-generatedTextCorpus) (LorenzM97, public GitHub dataset)

A balanced, English-language subset of **400 documents** was used:

| Source        | Class        | # Texts |
|---------------|--------------|---------|
| News articles | Human        | 100     |
| News articles | AI (GPT-3.5) | 100     |
| Wikipedia     | Human        | 100     |
| Wikipedia     | AI (GPT-3.5) | 100     |

Only *purely AI-generated* texts (created from a title prompt) were kept for the AI class — not the *rephrased* variants — so the two classes stay clean and comparable. The dataset is perfectly balanced (200/200) to avoid classifier bias.

**Fields:** `text`, `title`, `domain` (news/wiki), `language`, `label` (Human/AI)

---

## 🔧 Methodology / Pipeline

| Step | Description |
|------|-------------|
| 1. Problem Definition | Binary classification: Human vs. AI |
| 2. Data Collection | Balanced English corpus (news + Wikipedia, 400 docs) |
| 3. Preprocessing | Lowercasing, URL/digit removal, whitespace cleanup, deduplication |
| 4. Feature Engineering | Linguistic stats (word count, avg. word length, lexical diversity) + TF-IDF (unigrams & bigrams, 3000 features) |
| 5. Modeling | Naive Bayes, Logistic Regression, Linear SVM |
| 6. Evaluation | 80/20 stratified train/test split + 5-fold cross-validation |

This project applies **Classification**, one of the core Data Mining techniques (alongside Clustering, Association Rules, Regression, and Outlier Detection).

---

## 🤖 Algorithms Compared

| Algorithm | How it works | Why used |
|---|---|---|
| **Naive Bayes** | Probabilistic; assumes word independence | Fast, common baseline for text |
| **Logistic Regression** | Linear model with interpretable coefficients | Reveals which words drive each class |
| **Linear SVM** | Finds the max-margin decision boundary | Strong on high-dimensional, sparse TF-IDF data |

All three were trained and compared on identical data for an evidence-based choice of the best model, rather than assuming one algorithm is universally best.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-score | 5-fold CV F1 |
|---|---|---|---|---|---|
| Naive Bayes | 51.3% | 51.2% | 55.0% | 0.530 | 0.478 |
| Logistic Regression | 58.8% | 57.8% | 65.0% | 0.612 | 0.564 |
| **Linear SVM (best)** | **66.3%** | **65.9%** | **67.5%** | **0.667** | **0.683** |

**Confusion Matrix (Linear SVM, test set n=80):** 26 True Human, 27 True AI, 13–14 misclassifications each direction — errors are balanced, meaning the model isn't biased toward either class.

**Most indicative terms** (from Logistic Regression coefficients):
- → AI: *despite, significant, challenges, impact, government, continue...*
- → Human: *said, told, million, says, called, state...*

### Comparison with Literature

| Study | Approach | Best Reported Accuracy |
|---|---|---|
| Bataineh et al. (2025) | BoW/TF-IDF/classical ML vs. BERT | ~90.8% (BERT) |
| arXiv 2404.10032 | SVM, XGBoost, BERT | SVM 81%, XGBoost 84%, BERT 93% |
| Scientific Reports (2025) | Classical ML vs. RoBERTa | Classical ~79–80%, RoBERTa 96.1% |
| arXiv 2510.22874 | Baseline classical detection | 58.35% |
| **This project** | TF-IDF + Naive Bayes/LR/SVM | **66.3% (Linear SVM)** |

Classical ML methods typically reach **58–84%** accuracy, while transformer models (BERT/RoBERTa) reach **90–96%**. This project's result sits solidly within the classical-ML range, which is expected given the scope (classical algorithms, no deep learning).

---

## 🛠️ Tools & Libraries

- **Python 3**
- **pandas** — data loading & manipulation
- **scikit-learn** — TF-IDF vectorization, models, evaluation metrics, cross-validation
- **matplotlib** — visualizations

---

## 🚀 Getting Started

### Installation
```bash
pip install pandas scikit-learn matplotlib
```

### Usage
```bash
python ai_text_detection.py
```

This will print dataset stats and model comparison results to the console, and generate:
- `confusion_matrix.png` — confusion matrix of the best model
- `model_comparison.png` — bar chart comparing all models
- `model_comparison_results.csv` — results table
- `top_terms.txt` — most indicative words per class

---

## 📁 Project Structure

```
.
├── ai_text_detection.py           # Main pipeline: preprocessing → features → training → evaluation
├── ai_vs_human_text.csv           # Dataset (400 balanced documents)
├── model_comparison_results.csv   # Output: metrics per model
├── confusion_matrix.png           # Output: confusion matrix visualization
├── model_comparison.png           # Output: model comparison chart
└── README.md
```

---

## 🔮 Future Work

- Expand the dataset (more documents, more domains, multiple AI models such as GPT-4, Claude, Gemini)
- Add richer features: perplexity, burstiness, punctuation patterns, POS-tag distributions
- Experiment with deep learning approaches (fine-tuned BERT/RoBERTa) for a stronger baseline
- Test robustness against adversarial AI text that has been lightly paraphrased to evade detection

---

## 📚 References

- LorenzM97 (2023). *Human-AI-Generated Text Corpus* [Dataset]. GitHub.
- Bataineh, A. A. et al. (2025). *AI-Generated vs. Human Text: Introducing a New Dataset for Benchmarking and Analysis.*
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825-2830.
- Han, J., Kamber, M., & Pei, J. *Data Mining: Concepts and Techniques* (3rd ed.).

---

## 👤 Author

**Reham Said** — Data Mining course project
