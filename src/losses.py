"""Loss helpers for LDNets experiments."""

import tensorflow as tf


def reconstruction_loss(prediction, target):
    return tf.reduce_mean(tf.square(prediction - target))


def jepa_loss(predicted_embedding, target_embedding):
    return tf.reduce_mean(tf.square(predicted_embedding - tf.stop_gradient(target_embedding)))


def dynamics_consistency_loss(rollout_latent, teacher_latent):
    return tf.reduce_mean(tf.square(rollout_latent - tf.stop_gradient(teacher_latent)))


def latent_smoothness_loss(states):
    if states.shape[1] == 1:
        return tf.constant(0.0, dtype=states.dtype)
    return tf.reduce_mean(tf.square(states[:, 1:, :] - states[:, :-1, :]))


def weight_l2(networks):
    kernels = []
    for network in networks:
        for variable in network.trainable_variables:
            if "kernel" in variable.name:
                kernels.append(tf.reduce_mean(tf.square(variable)))
    if not kernels:
        return tf.constant(0.0, dtype=tf.float64)
    return tf.add_n(kernels) / len(kernels)
