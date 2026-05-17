# Project Report: Calendar Date Sequence Generation with a Language Model

---

## 1. Executive Summary & Methodology

This project addresses a structured **sequence generation** task: given a set of date attributes (day of week, month, leap year status, and a century code), train a language model capable of generating the corresponding full date string.

The dataset (`data.txt`) contains records spanning from 1800 to the early 2100s, each encoded in a structured token format:

```
[WED] [JAN] [False] [180] 1-1-1800
```

The model is trained on these records as a **next-token prediction** task — learning the statistical relationships between the structured prefix attributes and the target date output. The evaluation input (`example_input.txt`) strips the date suffix, providing only the structured tokens, and the model is expected to generate the correct date string.

The overall methodology follows a classic **autoregressive language modelling** pipeline: preprocess and tokenize the raw text, train the model with a standard sequence prediction objective, and evaluate output quality by comparing generated dates against ground truth.

---

## 2. Design Choices & Reflection (Why It Works)

**Why structured tokenization?**
The data uses a consistent bracket-delimited format (`[DAY]`, `[MONTH]`, `[True/False]`, `[YYY]`), which makes the vocabulary small and deterministic. This regularizes the tokenization space and prevents the model from wasting capacity on irregular characters or formatting noise.

**Why an autoregressive sequence model?**
Calendar dates are inherently rule-governed — leap year cycles, month boundaries, day-of-week progressions. An autoregressive model, trained token-by-token with teacher forcing, naturally learns these cyclic and conditional patterns without needing hand-crafted rules.

**Why this architecture works for this problem:**
The input data's structure creates strong conditional dependencies. For instance, knowing a year is a leap year (`[True]`) constrains which months can have 29 days, and knowing the century code (`[180]`) directly encodes the decade. These are *learnable* patterns, and a model with sufficient sequence memory can internalize them — resulting in consistent, rule-compliant date outputs.

**Why address class imbalance?**
The calendar date dataset contains unequal representation across attribute combinations — months with 31 days contribute more training records than February, and certain day-of-week patterns recur more frequently across common year ranges. Without correction, a model trained on this imbalanced data risks biasing toward majority patterns and underperforming on rare but valid dates (e.g., February 29th on a leap year, or dates near century boundaries). To counter this, class-weighted sampling was applied during training, upweighting underrepresented classes so the model is exposed to minority cases at sufficient frequency to learn them reliably — rather than exploiting majority-class shortcuts.

---

## 3. Core Technical Components (Explained Simply)

### Tokenization

The raw text is tokenized using the natural bracket-delimited units already present in the data. Each unique token — such as `[WED]`, `[JAN]`, `[True]`, `[180]`, `1`, `-`, `1800` — is assigned an integer index. The full vocabulary size is small (under ~100 tokens), keeping the embedding layer lightweight.

During training, each full record is treated as a single sequence. The model receives the structured attributes as context and learns to output the date characters one token at a time.

### Architecture

Two architectures were explored for this task, each approaching the sequence modelling problem differently.

---

#### Architecture 1: LSTM (Long Short-Term Memory)

The LSTM is a **recurrent neural network** that processes the input sequence one token at a time, maintaining a hidden state that carries information forward across the sequence.

1. **Embedding Layer** — each integer token is mapped to a dense vector of fixed size, giving the model a continuous representation of each input symbol.
2. **LSTM Layers** — one or more stacked LSTM layers process the token embeddings left-to-right. At each step, the LSTM updates its **hidden state** and **cell state**, which act as a memory — allowing the model to remember earlier tokens (e.g., the leap year flag) when predicting later ones (e.g., the day of the month).
3. **Output Projection** — a linear layer maps the final hidden state at each position to a probability distribution over the vocabulary, from which the next token is predicted.

**How information flows:** the model reads `[WED]`, then `[JAN]`, then `[False]`, then `[180]` sequentially, updating its internal memory at each step. By the time it reaches the date output, all four attributes are encoded in the hidden state and the model generates `1-1-1800` one character at a time.

**Strengths for this task:** LSTMs are well-suited to short, structured sequences where order matters. The gating mechanism (input, forget, output gates) helps the model selectively retain critical information like the leap year flag across many tokens.

---

#### Architecture 2: Transformer Decoder

The Transformer takes a fundamentally different approach — instead of processing tokens one by one, it attends to **all tokens in the sequence simultaneously** using self-attention.

1. **Embedding Layer + Positional Encoding** — each token is embedded into a dense vector, and a positional encoding is added to give the model a sense of token order (since Transformers have no built-in notion of sequence position).
2. **Masked Self-Attention Layers** — each token attends to all *previous* tokens in the sequence (masking future tokens to preserve the autoregressive property). This allows the model to directly relate, for example, the leap year token `[True]` to the month token `[FEB]` in a single step, without needing to carry that information through intermediate hidden states.
3. **Feed-Forward Layers** — a position-wise feed-forward network processes each token's representation independently after attention.
4. **Output Projection** — a linear layer maps each token's final representation to logits over the vocabulary.

**How information flows:** rather than a sequential left-to-right pass, the Transformer computes relationships between all tokens in parallel. The attention mechanism learns *which* tokens to focus on — for instance, when generating the year digits, it can attend directly back to the century code `[180]` regardless of how many tokens came in between.

**Strengths for this task:** Transformers handle long-range dependencies more efficiently than LSTMs. For this dataset, where the century code and leap year flag must be precisely linked to the output date, direct attention provides a clear advantage over relying on a compressed hidden state.

---

#### Architecture 3: CGAN (Conditional Generative Adversarial Network)

A CGAN extends the classic GAN framework by conditioning both the Generator and Discriminator on auxiliary information — in this case, the structured date attributes (`[DAY]`, `[MONTH]`, `[LEAP_YEAR]`, `[YEAR_CODE]`).

The architecture consists of two competing networks trained simultaneously:

1. **Generator** — takes a random noise vector $z$ concatenated with the condition embedding (the four date attributes) as input. It passes this through a series of fully connected or recurrent layers and learns to produce a plausible date string that is consistent with the given condition.
2. **Discriminator** — receives a date string (either real from `data.txt` or generated by the Generator) alongside the same condition embedding, and learns to distinguish real date–condition pairs from fake ones.

**How information flows:** the condition is embedded and injected at the input of both networks. The Generator is forced to produce outputs that satisfy the condition, because the Discriminator specifically checks whether the generated date is *consistent* with the provided attributes — not just whether it looks like a date in general.

**Training objective:** the Generator and Discriminator play a minimax game:

$$\min_G \max_D \; \mathbb{E}[\log D(x \mid c)] + \mathbb{E}[\log(1 - D(G(z \mid c) \mid c))]$$

where $c$ is the condition vector, $x$ is a real date string, and $z$ is sampled noise.

**Strengths for this task:** CGANs excel at learning to generate outputs conditioned on structured metadata. The adversarial training signal pushes the Generator beyond simple token probability matching — the Discriminator enforces global plausibility, meaning a generated date must be fully coherent with the given day, month, and year attributes to fool it.

**Loss Penalty and Training Balance:**
A known failure mode in GAN training is the Discriminator becoming too strong too quickly. Once it can perfectly separate real from fake examples, the gradients it sends back to the Generator vanish — the Generator receives no useful feedback and stops improving. In the extreme, this leads to **mode collapse**, where the Generator collapses to producing a narrow set of outputs (e.g., always the same date format) rather than learning the full target distribution.

To prevent this, we increased the **penalty parameter**: a regularisation term added to the Discriminator's training objective that limits how aggressively it can update relative to the Generator. By penalising the Discriminator for growing too confident too fast, the penalty keeps the adversarial dynamic productive — the Generator continues to receive meaningful gradient signal throughout training, both networks improve together, and overall training stability is preserved.

---

#### Architecture 4: CVAE (Conditional Variational Autoencoder)

A CVAE is a probabilistic generative model that learns a **latent space** of date representations, conditioned on the structured attributes. Unlike the GAN's adversarial setup, the CVAE trains via reconstruction and regularisation losses simultaneously.

The architecture has two components:

1. **Encoder** — takes the full date string (input + target) concatenated with the condition embedding and maps it to a probability distribution in latent space, parameterised by a mean $\mu$ and log-variance $\log \sigma^2$. A latent vector $z$ is sampled from this distribution using the reparameterisation trick: $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.
2. **Decoder** — takes the sampled latent vector $z$ concatenated with the condition embedding and reconstructs the original date string token by token.

**How information flows:** during training, the Encoder compresses a real date into a compact latent representation that captures its essential structure, conditioned on the known attributes. The Decoder then reconstructs it. At inference time, the Encoder is discarded — a latent vector is sampled directly from the prior $\mathcal{N}(0, I)$ and passed to the Decoder with the desired condition to generate a new date.

**Training objective:** the CVAE minimises the Evidence Lower Bound (ELBO):

$$\mathcal{L} = \underbrace{\mathbb{E}[\log p(x \mid z, c)]}_{\text{Reconstruction Loss}} - \underbrace{D_{KL}(q(z \mid x, c) \,\|\, p(z))}_{\text{KL Divergence}}$$

The reconstruction term encourages accurate date generation; the KL divergence term regularises the latent space to stay close to a standard Gaussian, ensuring smooth and generalisable sampling.

**Strengths for this task:** CVAEs provide a principled probabilistic framework for conditional generation. The structured latent space allows the model to interpolate between date representations and generate diverse yet valid outputs for a given condition — useful for exploring the full range of dates consistent with a set of attributes.

### Loss Function

**Cross-Entropy Loss** is used, computed at each predicted token position against the ground-truth token.

**Why cross-entropy?** Because next-token prediction is a classification problem — at every step, the model picks one token from a fixed vocabulary. Cross-entropy directly penalizes the model for assigning low probability to the correct token, creating a strong gradient signal that pushes predictions toward the right characters.

Formally, for a sequence of length $T$:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(\hat{y}_t = y_t)$$

This loss is computed over the date portion of each record (the output side), not the structured attribute prefix (the input/context side).

---

## 4. Training Performance & Output Analysis

### Loss Curves

> **[Insert: Training and Validation Loss Curves over epochs]**
>
> *Expected interpretation: Training loss should show a smooth downward curve converging within 20–50 epochs. Validation loss should closely track training loss given the highly regular, rule-governed nature of the data. A large gap between the two (validation loss rising while training loss falls) would indicate overfitting — more likely if the model is oversized relative to the small, finite vocabulary of this dataset. A plateau in both curves indicates successful convergence.*

### Output Quality

> **[Insert: Sample generated dates vs. ground-truth dates from `example_input.txt`]**
>
> **[Insert: Token-level accuracy or exact-match accuracy table across the test set]**
>
> *Expected interpretation: High token-level accuracy (>90%) is expected on held-out data given the deterministic nature of calendar rules. Failures are most likely on edge cases — century-boundary leap year exceptions (e.g., 1900 is not a leap year despite being divisible by 4) or day-of-week transitions across month/year boundaries. These failure modes are informative: they reveal whether the model has genuinely learned the Gregorian calendar rules, or has memorized common patterns without generalizing to rare edge cases.*

### Key Observations

- The model benefits directly from the structured input format: because day, month, leap year, and century code are *explicitly* provided as tokens, the generation task reduces to a structured lookup rather than open-ended generation.
- Low vocabulary size and high data regularity means the model should converge quickly and stably with minimal hyperparameter tuning.
- Any observed instability in training (oscillating loss) would suggest a learning rate that is too high for this low-variance dataset.

---

*Report generated for Assignment 2 — Generative AI, Semester 2.*
