import tensorflow as tf

class ConditionEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super(ConditionEmbedding, self).__init__()
        self.embed_day = tf.keras.layers.Embedding(7, 8)
        self.embed_month = tf.keras.layers.Embedding(12, 8)
        self.embed_leap = tf.keras.layers.Embedding(2, 4)
        self.embed_decade = tf.keras.layers.Embedding(41, 16)

    def call(self, inputs):
        e_day = self.embed_day(inputs[:, 0])
        e_month = self.embed_month(inputs[:, 1])
        e_leap = self.embed_leap(inputs[:, 2])
        e_decade = self.embed_decade(inputs[:, 3])
        return tf.concat([e_day, e_month, e_leap, e_decade], axis=-1)

# 1. Seq2Seq (Encoder-Decoder)
class Seq2SeqModel(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Seq2SeqModel, self).__init__()
        self.cond_embed = ConditionEmbedding()
        self.target_embed = tf.keras.layers.Embedding(vocab_size, 16)
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(vocab_size)
        
        self.state_h_dense = tf.keras.layers.Dense(64)
        self.state_c_dense = tf.keras.layers.Dense(64)

    def call(self, inputs, targets=None, training=False):
        context = self.cond_embed(inputs)
        state_h = self.state_h_dense(context)
        state_c = self.state_c_dense(context)
        
        batch_size = tf.shape(inputs)[0]
        
        if training and targets is not None:
            # Prepend <SOS> token (ID: 32)
            sos = tf.ones((batch_size, 1), dtype=tf.int32) * 32
            dec_input_seq = tf.concat([sos, targets[:, :-1]], axis=1)
            dec_input = self.target_embed(dec_input_seq)
            lstm_out, _, _ = self.lstm(dec_input, initial_state=(state_h, state_c), training=training)
            lstm_out = self.dropout(lstm_out, training=training)
            return self.dense(lstm_out)
        else:
            current_input = self.target_embed(tf.ones((batch_size, 1), dtype=tf.int32) * 32) 
            states = (state_h, state_c)
            predictions = []
            for _ in range(3): # Sequence is exactly 3 tokens long!
                lstm_out, h, c = self.lstm(current_input, initial_state=states, training=training)
                states = (h, c)
                logits = self.dense(lstm_out)
                predicted_id = tf.argmax(logits, axis=-1, output_type=tf.int32)
                predictions.append(predicted_id)
                current_input = self.target_embed(predicted_id)
            return tf.concat(predictions, axis=1)

# 2. Conditional GAN (cGAN)
class CGAN_Generator(tf.keras.Model):
    def __init__(self, vocab_size, seq_len=3):
        super(CGAN_Generator, self).__init__()
        self.cond_embed = ConditionEmbedding()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(seq_len * vocab_size)
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def call(self, inputs, noise, training=False):
        cond = self.cond_embed(inputs)
        x = tf.concat([cond, noise], axis=-1)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return tf.reshape(x, (-1, self.seq_len, self.vocab_size))

class CGAN_Discriminator(tf.keras.Model):
    def __init__(self, vocab_size):
        super(CGAN_Discriminator, self).__init__()
        self.cond_embed = ConditionEmbedding()
        self.seq_flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.out = tf.keras.layers.Dense(1) 

    def call(self, inputs, seq_one_hot, training=False):
        cond = self.cond_embed(inputs)
        seq_flat = self.seq_flatten(seq_one_hot)
        x = tf.concat([cond, seq_flat], axis=-1)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.out(x)

# 3. Conditional VAE (cVAE)
class CVAE(tf.keras.Model):
    def __init__(self, vocab_size, latent_dim=16, seq_len=3):
        super(CVAE, self).__init__()
        self.cond_embed = ConditionEmbedding()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        self.enc_flat = tf.keras.layers.Flatten()
        self.enc_dense = tf.keras.layers.Dense(64, activation='relu')
        self.enc_drop = tf.keras.layers.Dropout(0.2)
        self.enc_mu = tf.keras.layers.Dense(latent_dim)
        self.enc_logvar = tf.keras.layers.Dense(latent_dim)
        
        self.dec_dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dec_drop = tf.keras.layers.Dropout(0.2)
        self.dec_out = tf.keras.layers.Dense(seq_len * vocab_size)

    def encode(self, inputs, seq_one_hot, training=False):
        cond = self.cond_embed(inputs)
        seq_flat = self.enc_flat(seq_one_hot)
        x = tf.concat([cond, seq_flat], axis=-1)
        x = self.enc_dense(x)
        x = self.enc_drop(x, training=training)
        return self.enc_mu(x), self.enc_logvar(x)

    def decode(self, inputs, z, training=False):
        cond = self.cond_embed(inputs)
        x = tf.concat([cond, z], axis=-1)
        x = self.dec_dense1(x)
        x = self.dec_drop(x, training=training)
        x = self.dec_out(x)
        return tf.reshape(x, (-1, self.seq_len, self.vocab_size))

    def call(self, inputs, seq_one_hot=None, training=False):
        if seq_one_hot is None:
            seq_one_hot = tf.zeros((tf.shape(inputs)[0], self.seq_len, self.vocab_size))
        mu, logvar = self.encode(inputs, seq_one_hot, training=training)
        eps = tf.random.normal(shape=tf.shape(mu))
        z = mu + tf.exp(logvar * 0.5) * eps
        return self.decode(inputs, z, training=training)

# --- Transformer Components ---
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 4. Transformer One-Shot
class TransformerOneShotModel(tf.keras.Model):
    def __init__(self, vocab_size, seq_len=3, embed_dim=36, num_heads=2, ff_dim=64):
        super(TransformerOneShotModel, self).__init__()
        self.cond_embed = ConditionEmbedding() 
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        self.pos_emb = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dense_out = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        cond = self.cond_embed(inputs)
        cond_expanded = tf.repeat(tf.expand_dims(cond, 1), self.seq_len, axis=1)
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        pos_embeddings = self.pos_emb(positions) 
        
        x = cond_expanded + pos_embeddings
        x = self.transformer_block(x, training=training)
        return self.dense_out(x)