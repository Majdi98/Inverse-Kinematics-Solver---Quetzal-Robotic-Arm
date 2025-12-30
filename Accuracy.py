import tensorflow as tf
import numpy as np

# Paper thresholds (Table 9) 
MSE_THRESHOLD = 0.03    # MSE < 0.03
MAE_THRESHOLD = 0.05    # MAE ≤ 0.05
R2_THRESHOLD = 0.9      # R² ≥ 0.9
ED_THRESHOLD = 0.5      # Euclidean Distance ≤ 0.5

def paper_accuracy(y_true, y_pred):
    # per-sample MSE & MAE
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)

    # per-sample R²
    y_mean = tf.reduce_mean(y_true, axis=1, keepdims=True)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=1)
    ss_tot = tf.reduce_sum(tf.square(y_true - y_mean), axis=1)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-7))

    # Euclidean distance
    ed = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

    meets_all = (
        (mse < MSE_THRESHOLD) &
        (mae <= MAE_THRESHOLD) &
        (r2 >= R2_THRESHOLD) &
        (ed <= ED_THRESHOLD)
    )

    return tf.reduce_mean(tf.cast(meets_all, tf.float32)) * 100

def custom_accuracy(y_true, y_pred):
    """
    Custom accuracy Made for regression, metric (1 - normalized MAE)
    for comparison.
    """
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    max_error = tf.constant(np.pi, dtype=tf.float32)
    normalized_accuracy = 1.0 - (mae / max_error)
    return normalized_accuracy

def custom_accuracy2(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1) 
    return mae <= 0.05  # % of “close enough” predictions
