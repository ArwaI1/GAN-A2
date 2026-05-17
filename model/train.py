import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple
from data_utils import create_datasets, TARGET_VOCAB_SIZE
from architecture import Seq2SeqModel, CGAN_Generator, CGAN_Discriminator, CVAE, TransformerOneShotModel

# BEST PRACTICE: Explicit manual seed configuration for experiment replication
SEED: int = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

vocab_size: int = TARGET_VOCAB_SIZE
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

class EarlyStoppingTracker:
    """Tracks validation loss to prevent overfitting and safely stop training early."""
    def __init__(self, patience: int = 4, min_delta: float = 0.005) -> None:
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.best_loss: float = float('inf')
        self.wait: int = 0

    def check_stop(self, current_val_loss: float) -> bool:
        if current_val_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_val_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n--- Early Stopping Triggered (Validation loss stopped improving for {self.patience} epochs) ---")
                return True
            return False

def save_loss_plot(history: dict, filename: str, title: str) -> None:
    """Helper function to plot and save metrics to the plots/ directory."""
    plt.figure(figsize=(8, 5))
    for label, values in history.items():
        plt.plot(range(1, len(values) + 1), values, label=label, lw=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    os.makedirs('plots', exist_ok=True)
    plot_path = os.path.join('plots', filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📈 Loss visualization plot successfully saved to: {plot_path}")

def train_seq2seq(train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, epochs: int) -> None:
    model = Seq2SeqModel(vocab_size)
    optimizer = tf.keras.optimizers.Adam()
    early_stop = EarlyStoppingTracker()
    
    history = {'Train Loss': [], 'Val Loss': []}
    
    @tf.function
    def train_step(X, Y):
        with tf.GradientTape() as tape:
            preds = model(X, targets=Y, training=True)
            loss = scce(Y, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(X, Y):
        preds = model(X, targets=Y, training=True)
        return scce(Y, preds)

    for epoch in range(epochs):
        train_loss, train_steps = 0.0, 0
        for X, Y in train_dataset:
            loss = train_step(X, Y)
            train_loss += loss.numpy()
            train_steps += 1
        
        val_loss, val_steps = 0.0, 0
        for X, Y in val_dataset:
            loss = val_step(X, Y)
            val_loss += loss.numpy()
            val_steps += 1
            
        avg_train: float = train_loss / train_steps
        avg_val: float = val_loss / val_steps
        
        history['Train Loss'].append(avg_train)
        history['Val Loss'].append(avg_val)
        
        print(f"Seq2Seq Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        if early_stop.check_stop(avg_val): break

    model.save_weights('weights/seq2seq.weights.h5')
    save_loss_plot(history, 'seq2seq_loss.png', 'Seq2Seq Model Training History')

def train_gan(train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, epochs: int) -> None:
    gen = CGAN_Generator(vocab_size)
    disc = CGAN_Discriminator(vocab_size)
    
    # FIX: Implemented balanced learning rates & beta parameters to stabilize training
    gen_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    disc_opt = tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5)

    history = {
        'Train Gen Loss': [], 'Train Disc Loss': [],
        'Val Gen Loss': [], 'Val Disc Loss': []
    }

    @tf.function
    def train_step(X, Y):
        real_one_hot = tf.one_hot(Y, vocab_size)
        batch_size = tf.shape(X)[0]
        noise = tf.random.normal([batch_size, 16])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_logits = gen(X, noise, training=True)
            fake_one_hot = tf.nn.softmax(fake_logits)
            
            real_output = disc(X, real_one_hot, training=True)
            fake_output = disc(X, fake_one_hot, training=True)

            disc_loss = bce(tf.ones_like(real_output) * 0.9, real_output) + bce(tf.zeros_like(fake_output), fake_output)
            
            # FIX: Adding an auxiliary categorical cross-entropy anchor penalty to the generator loss
            # This prevents it from producing unguided random tokens that the discriminator easily catches.
            adv_gen_loss = bce(tf.ones_like(fake_output), fake_output)
            aux_recon_loss = scce(Y, fake_logits)
            gen_loss = adv_gen_loss + (1.5 * aux_recon_loss)

        gen_grads = gen_tape.gradient(gen_loss, gen.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, disc.trainable_variables)
        gen_opt.apply_gradients(zip(gen_grads, gen.trainable_variables))
        disc_opt.apply_gradients(zip(disc_grads, disc.trainable_variables))
        return gen_loss, disc_loss

    @tf.function
    def val_step(X, Y):
        real_one_hot = tf.one_hot(Y, vocab_size)
        batch_size = tf.shape(X)[0]
        noise = tf.random.normal([batch_size, 16])
        fake_logits = gen(X, noise, training=False)
        fake_one_hot = tf.nn.softmax(fake_logits)
        real_output = disc(X, real_one_hot, training=False)
        fake_output = disc(X, fake_one_hot, training=False)
        
        d_loss = bce(tf.ones_like(real_output), real_output) + bce(tf.zeros_like(fake_output), fake_output)
        g_loss = bce(tf.ones_like(fake_output), fake_output) + (1.5 * scce(Y, fake_logits))
        return g_loss, d_loss

    for epoch in range(epochs):
        g_train, d_train, steps = 0.0, 0.0, 0
        for X, Y in train_dataset:
            g_loss, d_loss = train_step(X, Y)
            g_train += g_loss.numpy()
            d_train += d_loss.numpy()
            steps += 1
            
        g_val, d_val, val_steps = 0.0, 0.0, 0
        for X, Y in val_dataset:
            gv, dv = val_step(X, Y)
            g_val += gv.numpy()
            d_val += dv.numpy()
            val_steps += 1
            
        avg_g_train = g_train / steps
        avg_d_train = d_train / steps
        avg_g_val = g_val / val_steps
        avg_d_val = d_val / val_steps

        history['Train Gen Loss'].append(avg_g_train)
        history['Train Disc Loss'].append(avg_d_train)
        history['Val Gen Loss'].append(avg_g_val)
        history['Val Disc Loss'].append(avg_d_val)
        
        print(f"GAN Epoch {epoch+1:02d} | Train G/D Loss: {avg_g_train:.3f}/{avg_d_train:.3f} | Val G/D Loss: {avg_g_val:.3f}/{avg_d_val:.3f}")
        
    gen.save_weights('weights/cgan_gen.weights.h5')
    save_loss_plot(history, 'cgan_loss.png', 'Conditional GAN Training History')

def train_vae(train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, epochs: int) -> None:
    model = CVAE(vocab_size)
    optimizer = tf.keras.optimizers.Adam()
    early_stop = EarlyStoppingTracker()

    history = {'Train Loss': [], 'Val Loss': []}

    @tf.function
    def train_step(X, Y):
        real_one_hot = tf.one_hot(Y, vocab_size)
        with tf.GradientTape() as tape:
            mu, logvar = model.encode(X, real_one_hot, training=True)
            eps = tf.random.normal(shape=tf.shape(mu))
            z = mu + tf.exp(logvar * 0.5) * eps
            logits = model.decode(X, z, training=True)
            
            recon_loss = scce(Y, logits)
            kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
            loss = recon_loss + kl_loss
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(X, Y):
        real_one_hot = tf.one_hot(Y, vocab_size)
        mu, logvar = model.encode(X, real_one_hot, training=False)
        eps = tf.random.normal(shape=tf.shape(mu))
        z = mu + tf.exp(logvar * 0.5) * eps
        logits = model.decode(X, z, training=False)
        recon_loss = scce(Y, logits)
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        return recon_loss + kl_loss

    for epoch in range(epochs):
        train_loss, steps = 0.0, 0
        for X, Y in train_dataset:
            loss = train_step(X, Y)
            train_loss += loss.numpy()
            steps += 1
            
        val_loss, val_steps = 0.0, 0
        for X, Y in val_dataset:
            loss = val_step(X, Y)
            val_loss += loss.numpy()
            val_steps += 1
            
        avg_train: float = train_loss / steps
        avg_val: float = val_loss / val_steps
        
        history['Train Loss'].append(avg_train)
        history['Val Loss'].append(avg_val)
        
        print(f"VAE Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        if early_stop.check_stop(avg_val): break

    dummy_x = tf.zeros((1, 4), dtype=tf.int32)
    model(dummy_x) 
    model.save_weights('weights/cvae.weights.h5')
    save_loss_plot(history, 'cvae_loss.png', 'Conditional VAE Training History')

def train_transformer(train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, epochs: int) -> None:
    model = TransformerOneShotModel(vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    early_stop = EarlyStoppingTracker()

    history = {'Train Loss': [], 'Val Loss': []}

    @tf.function
    def train_step(X, Y):
        with tf.GradientTape() as tape:
            logits = model(X, training=True) 
            loss = scce(Y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(X, Y):
        logits = model(X, training=False)
        return scce(Y, logits)

    for epoch in range(epochs):
        train_loss, steps = 0.0, 0
        for X, Y in train_dataset:
            loss = train_step(X, Y)
            train_loss += loss.numpy()
            steps += 1
            
        val_loss, val_steps = 0.0, 0
        for X, Y in val_dataset:
            loss = val_step(X, Y)
            val_loss += loss.numpy()
            val_steps += 1
            
        avg_train: float = train_loss / steps
        avg_val: float = val_loss / val_steps
        
        history['Train Loss'].append(avg_train)
        history['Val Loss'].append(avg_val)
        
        print(f"Transformer Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        if early_stop.check_stop(avg_val): break

    model.save_weights('weights/transformer.weights.h5')
    save_loss_plot(history, 'transformer_loss.png', 'Transformer One-Shot Training History')

if __name__ == "__main__":
    os.makedirs('weights', exist_ok=True)
    
    current_dir: str = os.path.dirname(os.path.abspath(__file__))
    data_path: str = os.path.join(current_dir, '..', 'data', 'data.txt')
    
    print(f"Loading split data from: {data_path}")
    train_set, val_set = create_datasets(data_path, batch_size=64)
    
    print("Select a model to train:")
    print("1. Seq2Seq (Encoder-Decoder)")
    print("2. Conditional GAN (cGAN)")
    print("3. Conditional VAE (cVAE)")
    print("4. Transformer (One-Shot)")
    choice: str = input("Enter 1, 2, 3, or 4: ")
    
    max_epochs: int = 40 
    if choice == '1': train_seq2seq(train_set, val_set, max_epochs)
    elif choice == '2': train_gan(train_set, val_set, max_epochs)
    elif choice == '3': train_vae(train_set, val_set, max_epochs)
    elif choice == '4': train_transformer(train_set, val_set, max_epochs)
    else: print("Invalid choice.")