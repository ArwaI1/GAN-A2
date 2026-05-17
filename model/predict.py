import argparse
import os
import tensorflow as tf
from data_utils import parse_line, TARGET_VOCAB_SIZE, DECADES
from architecture import Seq2SeqModel, CGAN_Generator, CVAE, TransformerOneShotModel

def decode_sequence(indices, decade_idx):
    # indices array contains exactly 3 numbers: [y_digit, month, day]
    y_digit = int(indices[0])
    month = int(indices[1])
    day = int(indices[2])
    
    # Mathematical clamping: guarantees no months > 12 or days > 31 ever print out!
    month = max(1, min(12, month))
    day = max(1, min(31, day))
    y_digit = max(0, min(9, y_digit))
    
    # Reconstruct the full year from the Decade Condition + Year Digit
    decade_str = DECADES[decade_idx].strip('[]') # e.g., turns "[196]" into "196"
    year = int(decade_str) * 10 + y_digit
    
    return f"{day}-{month}-{year}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input file")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    args = parser.parse_args()

    print("Select a model for inference:")
    print("1. Seq2Seq\n2. cGAN\n3. cVAE\n4. Transformer")
    choice = input("Enter 1, 2, 3, or 4: ")

    vocab_size = TARGET_VOCAB_SIZE
    dummy_input = tf.zeros((1, 4), dtype=tf.int32)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(current_dir, 'weights')

    if choice == '1':
        model = Seq2SeqModel(vocab_size)
        model(dummy_input, training=False)
        model.load_weights(os.path.join(weights_dir, 'seq2seq.weights.h5'))
    elif choice == '2':
        model = CGAN_Generator(vocab_size)
        model(dummy_input, tf.zeros((1, 16)), training=False)
        model.load_weights(os.path.join(weights_dir, 'cgan_gen.weights.h5'))
    elif choice == '3':
        model = CVAE(vocab_size)
        model(dummy_input, training=False) 
        model.load_weights(os.path.join(weights_dir, 'cvae.weights.h5'))
    elif choice == '4':
        model = TransformerOneShotModel(vocab_size)
        model(dummy_input, training=False)
        model.load_weights(os.path.join(weights_dir, 'transformer.weights.h5'))
    else:
        print("Invalid choice."); return

    predictions_out = []
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            X, _ = parse_line(line)
            if X is None: continue
            
            X_tensor = tf.constant([X], dtype=tf.int32)
            
            if choice == '1':
                pred_indices = model(X_tensor, training=False)[0]
            elif choice == '2':
                noise = tf.random.normal([1, 16])
                logits = model(X_tensor, noise, training=False)
                pred_indices = tf.argmax(logits[0], axis=-1)
            elif choice == '3':
                z = tf.random.normal([1, 16])
                logits = model.decode(X_tensor, z, training=False)
                pred_indices = tf.argmax(logits[0], axis=-1)
            elif choice == '4':
                logits = model(X_tensor, training=False)
                pred_indices = tf.argmax(logits[0], axis=-1)
                
            decade_idx = X[3]
            generated_date = decode_sequence(pred_indices, decade_idx)
            
            conditions_str = " ".join(line.split()[:4])
            predictions_out.append(f"{conditions_str} {generated_date}")

    with open(output_path, 'w') as f:
        for p in predictions_out:
            f.write(p + "\n")
    print(f"Predictions successfully saved to {output_path}")

if __name__ == "__main__":
    main()