import tensorflow as tf
import numpy as np
import os
from typing import Tuple, List, Optional

# BEST PRACTICE: Manual seed to allow for experiment replication!
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DAYS = ["[MON]", "[TUE]", "[WED]", "[THU]", "[FRI]", "[SAT]", "[SUN]"]
MONTHS = ["[JAN]", "[FEB]", "[MAR]", "[APR]", "[MAY]", "[JUN]", "[JUL]", "[AUG]", "[SEP]", "[OCT]", "[NOV]", "[DEC]"]
LEAP = ["[False]", "[True]"]
DECADES = [f"[{i}]" for i in range(180, 221)]

TARGET_VOCAB_SIZE = 33 

# BEST PRACTICE: Type Hinting added to function signatures
def parse_line(line: str) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    parts = line.strip().split()
    if len(parts) < 4: return None, None
    conditions = parts[:4]
    
    try:
        c_day = DAYS.index(conditions[0])
        c_month = MONTHS.index(conditions[1])
        c_leap = LEAP.index(conditions[2])
        c_decade = DECADES.index(conditions[3])
    except ValueError:
        return None, None
    
    X = [c_day, c_month, c_leap, c_decade]
    
    if len(parts) == 5:
        d, m, y = parts[4].split('-')
        Y = [int(y[-1]), int(m), int(d)]
        return X, Y
    else:
        return X, None

# BEST PRACTICE: Added validation split for testing unseen data
def create_datasets(filepath: str, batch_size: int = 32, val_split: float = 0.1) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    if not os.path.isabs(filepath):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filepath)
        
    X_data, Y_data = [], []
    with open(filepath, 'r') as f:
        for line in f:
            X, Y = parse_line(line)
            if X is not None and Y is not None:
                X_data.append(X)
                Y_data.append(Y)
                
                # Handling Leap Year Data Imbalance
                if X[2] == 1:
                    X_data.extend([X] * 3)
                    Y_data.extend([Y] * 3)

    X_data = np.array(X_data, dtype=np.int32)
    Y_data = np.array(Y_data, dtype=np.int32)
    
    # BEST PRACTICE: Shuffling with explicit seed
    dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data))
    dataset = dataset.shuffle(buffer_size=50000, seed=SEED)
    
    # Calculate split sizes
    total_samples = len(X_data)
    val_size = int(total_samples * val_split)
    
    # Split into Train and Validation
    val_dataset = dataset.take(val_size).batch(batch_size, drop_remainder=True)
    train_dataset = dataset.skip(val_size).batch(batch_size, drop_remainder=True)
    
    return train_dataset, val_dataset