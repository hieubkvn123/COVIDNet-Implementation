import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import BinaryCrossentropy

@tf.function
def train_step(model, optimizer, batch):
    bce = BinaryCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        images, labels = batch
        logits, logits_prob = model(images, training=True)

        loss = bce(labels, logits_prob)
        
        correct_prediction = tf.equal(tf.argmax(logits_prob, 1), tf.argmax(labels, 1))
        accuracy = tf.cast(correct_prediction, dtype=tf.float32)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return logits_prob, tf.reduce_mean(loss), tf.reduce_mean(accuracy)

@tf.function 
def val_step(model, batch):
    bce = BinaryCrossentropy(from_logits=True)

    images, labels = batch
    logits, logits_prob = model(images, training=False)

    loss = bce(labels, logits_prob)

    correct_prediction = tf.equal(tf.argmax(logits_prob, 1), tf.argmax(labels, 1))
    accuracy = tf.cast(correct_prediction, dtype=tf.float32)
    
    return tf.reduce_mean(loss), tf.reduce_mean(accuracy)
