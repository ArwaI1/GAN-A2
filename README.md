# Conditional Deep Generative Models for Structured Sequence Synthesis

An end-to-end deep learning project benchmarking four distinct generative paradigms for structured date sequence synthesis from conditional metadata attributes[cite: 14].

---

## 📌 Overview & Task Formulation

This project tackles a structured sequence generation task[cite: 14]: given a set of date attributes (day of week, month, leap year status, and a century code), train a neural model capable of generating the corresponding full date string[cite: 14].

### Input & Output Format
* **Raw Training Record**: `[WED] [JAN] [False] [180] 1-1-1800`[cite: 14]
* **Conditioning Attributes**: `[WED] [JAN] [False] [180]`[cite: 14]
* **Generated Output**: `1-1-1800`[cite: 14]

The dataset spans records from 1800 to the early 2100s encoded in a bracket-delimited token format[cite: 14].

---

## 🏗️ Model Architectures

Four comparative architectures were implemented and benchmarked[cite: 23]:

### 1. Seq2Seq (LSTM)
* **Approach**: Recurrent neural network with stacked LSTM layers processing tokens sequentially, updating hidden/cell states to retain long-range attribute context[cite: 15].
* **Loss Function**: Token-level Cross-Entropy Loss with teacher forcing[cite: 14, 18].

### 2. Transformer Decoder
* **Approach**: Autoregressive transformer with masked self-attention and positional encodings[cite: 16]. Attends directly across tokens in parallel to link distant attributes (e.g., century code and leap year flag) to output digits[cite: 16].

### 3. Conditional Variational Autoencoder (CVAE)
* **Approach**: Probabilistic generative model with an Encoder-Decoder framework mapping inputs into a latent distribution $z \sim \mathcal{N}(\mu, \sigma^2)$, conditioned on attribute embeddings[cite: 17].
* **Loss Function**: Evidence Lower Bound (ELBO) combining reconstruction cross-entropy loss and KL divergence regularization[cite: 18].

### 4. Conditional Generative Adversarial Network (CGAN)
* **Approach**: Adversarial minimax game between a Generator synthesizing date strings from noise $z$ and condition $c$, and a Discriminator evaluating real vs. fake date-condition pairs.
* **Regularization**: Added a Discriminator penalty parameter to restrict aggressive Discriminator updates, preventing vanishing gradients and early mode collapse[cite: 17].

---

## 📊 Benchmark Results

All models were evaluated on ground-truth validation records using exact string match accuracy[cite: 21, 23]:

| Architecture | Exact Match Accuracy | Epochs to Converge | Key Observation |
| :--- | :---: | :---: | :--- |
| **Transformer Decoder** | **78.82%**[cite: 23] | **7**[cite: 23] | Fast parallel convergence; stable validation loss (~2.38)[cite: 20, 23]. |
| **Seq2Seq (LSTM)** | **78.75%**[cite: 23] | **6**[cite: 23] | Winner; reliable sequence mapping with controlled generalization[cite: 19, 23]. |
| **Conditional VAE (CVAE)** | **75.38%**[cite: 23] | **17**[cite: 20, 23] | Smooth latent space convergence; minor stochastic sampling variance[cite: 20, 24]. |
| **Conditional GAN (CGAN)** | **54.23%**[cite: 23] | **40**[cite: 21, 23] | Discriminator dominated training, cutting off generator feedback[cite: 21, 24]. |

---

## 🔍 Key Insights & Analysis

* **Class Imbalance Handling**: Applied class-weighted sampling during training to upweight underrepresented attribute combinations (e.g., leap years, February dates), preventing bias toward majority patterns[cite: 15].
* **Primary Failure Mode**: Day-of-week mismatches were the dominant error across all models (e.g., predicting `SAT` instead of `WED`)[cite: 21, 24]. While neural models easily master static categorical rules (months, century codes), learning cyclical algorithmic logic (such as Zeller's Congruence) without explicit algorithmic supervision remains challenging[cite: 24].
* **LSTM vs. Transformer**: Achieved near-identical accuracy (~78.7%–78.8%), proving recurrent memory is highly competitive with self-attention for short, structured sequence tasks[cite: 24].

---

## 📁 Repository Structure

```text
.
├── data.txt                 # Dataset containing structured date records
├── example_input.txt        # Validation input with attributes only
├── train.py                 # Training script for all architectures
├── validate.py              # Validation script evaluating exact-match accuracy
├── models/                  # PyTorch model definitions (LSTM, Transformer, CVAE, CGAN)
└── README.md
