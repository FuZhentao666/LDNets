"""Shared model components for experimental LDNets variants."""

import tensorflow as tf


class PointSetEncoder(tf.keras.Model):
    """Encode an unordered set of coordinate/value observations."""

    def __init__(
        self,
        feature_dim,
        condition_dim,
        latent_dim,
        embedding_dim,
        width=32,
        depth=2,
        name="PointSetEncoder",
    ):
        super().__init__(name=name)
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        self.point_layers = [
            tf.keras.layers.Dense(width, activation=tf.nn.tanh)
            for _ in range(depth)
        ]
        self.context_layers = [
            tf.keras.layers.Dense(width, activation=tf.nn.tanh)
            for _ in range(depth)
        ]
        self.latent_head = tf.keras.layers.Dense(latent_dim)
        self.embedding_head = tf.keras.layers.Dense(embedding_dim)

    def call(self, point_features, condition=None):
        x = point_features
        for layer in self.point_layers:
            x = layer(x)
        pooled = tf.reduce_mean(x, axis=-2)

        if self.condition_dim > 0:
            if condition is None:
                batch = tf.shape(pooled)[0]
                condition = tf.zeros((batch, self.condition_dim), dtype=pooled.dtype)
            pooled = tf.concat([pooled, condition], axis=-1)

        for layer in self.context_layers:
            pooled = layer(pooled)

        return self.latent_head(pooled), self.embedding_head(pooled)


class LatentTransition(tf.keras.Model):
    """Euler latent transition used by JEPA-LDNet."""

    def __init__(
        self,
        latent_dim,
        condition_dim,
        width,
        depth=2,
        name="LatentTransition",
    ):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.layers_ = [
            tf.keras.layers.Dense(width, activation=tf.nn.tanh)
            for _ in range(depth)
        ]
        self.out = tf.keras.layers.Dense(latent_dim)

    def call(self, state, condition=None):
        if self.condition_dim > 0:
            if condition is None:
                batch = tf.shape(state)[0]
                condition = tf.zeros((batch, self.condition_dim), dtype=state.dtype)
            x = tf.concat([state, condition], axis=-1)
        else:
            x = state
        for layer in self.layers_:
            x = layer(x)
        return self.out(x)

    def euler_step(self, state, condition, dt_scale):
        return state + dt_scale * self(state, condition)


class ContinuousDecoder(tf.keras.Model):
    """Meshless coordinate-query decoder."""

    def __init__(
        self,
        latent_dim,
        space_dim,
        output_dim,
        width,
        depth=2,
        name="ContinuousDecoder",
    ):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.space_dim = space_dim
        self.output_dim = output_dim
        self.layers_ = [
            tf.keras.layers.Dense(width, activation=tf.nn.tanh)
            for _ in range(depth)
        ]
        self.out = tf.keras.layers.Dense(output_dim)

    def call(self, states, points_full):
        num_points = tf.shape(points_full)[2]
        states_expanded = tf.broadcast_to(
            tf.expand_dims(states, axis=2),
            [
                tf.shape(states)[0],
                tf.shape(states)[1],
                num_points,
                self.latent_dim,
            ],
        )
        x = tf.concat([states_expanded, points_full], axis=-1)
        for layer in self.layers_:
            x = layer(x)
        return self.out(x)


class JEPAPredictor(tf.keras.Model):
    """Predict target embeddings from rollout states."""

    def __init__(
        self,
        latent_dim,
        condition_dim,
        embedding_dim,
        width=32,
        depth=2,
        name="JEPAPredictor",
    ):
        super().__init__(name=name)
        self.condition_dim = condition_dim
        self.layers_ = [
            tf.keras.layers.Dense(width, activation=tf.nn.tanh)
            for _ in range(depth)
        ]
        self.out = tf.keras.layers.Dense(embedding_dim)

    def call(self, states_at_targets, target_times, condition=None, context_embedding=None):
        batch = tf.shape(states_at_targets)[0]
        num_targets = tf.shape(states_at_targets)[1]
        times = tf.broadcast_to(
            tf.reshape(target_times, (1, num_targets, 1)),
            (batch, num_targets, 1),
        )
        pieces = [states_at_targets, tf.cast(times, states_at_targets.dtype)]

        if self.condition_dim > 0:
            if condition is None:
                condition = tf.zeros((batch, self.condition_dim), dtype=states_at_targets.dtype)
            condition = tf.broadcast_to(
                tf.expand_dims(condition, axis=1),
                (batch, num_targets, self.condition_dim),
            )
            pieces.append(condition)

        if context_embedding is not None:
            context_embedding = tf.broadcast_to(
                tf.expand_dims(context_embedding, axis=1),
                (batch, num_targets, tf.shape(context_embedding)[-1]),
            )
            pieces.append(context_embedding)

        x = tf.concat(pieces, axis=-1)
        for layer in self.layers_:
            x = layer(x)
        return self.out(x)


class JEPALDNet(tf.keras.Model):
    """JEPA-constrained LDNet for sparse-observation latent rollout."""

    def __init__(
        self,
        feature_dim,
        condition_dim,
        latent_dim,
        embedding_dim,
        space_dim,
        output_dim,
        dynamics_width,
        reconstruction_width,
        encoder_width=32,
        predictor_width=32,
    ):
        super().__init__(name="JEPALDNet")
        self.context_encoder = PointSetEncoder(
            feature_dim,
            condition_dim,
            latent_dim,
            embedding_dim,
            width=encoder_width,
            name="ObservationEncoder_E_phi",
        )
        self.target_encoder = PointSetEncoder(
            feature_dim,
            condition_dim,
            latent_dim,
            embedding_dim,
            width=encoder_width,
            name="TargetEncoder_E_bar",
        )
        self.target_encoder.trainable = False
        self.transition = LatentTransition(
            latent_dim,
            condition_dim,
            dynamics_width,
            name="LatentTransition_T_theta",
        )
        self.decoder = ContinuousDecoder(
            latent_dim,
            space_dim,
            output_dim,
            reconstruction_width,
            name="ContinuousDecoder_D_omega",
        )
        self.predictor = JEPAPredictor(
            latent_dim,
            condition_dim,
            embedding_dim,
            width=predictor_width,
            name="Predictor_P_psi",
        )

    def encode_context(self, context_features, condition):
        z0, context_embedding = self.context_encoder(context_features, condition)
        return z0, context_embedding

    def rollout(self, z0, conditions, dt_scale):
        num_times = tf.shape(conditions)[1]
        states = tf.TensorArray(z0.dtype, size=num_times)
        state = z0
        states = states.write(0, state)
        for i in tf.range(num_times - 1):
            condition = conditions[:, i, :] if self.transition.condition_dim > 0 else None
            state = self.transition.euler_step(state, condition, dt_scale)
            states = states.write(i + 1, state)
        return tf.transpose(states.stack(), perm=(1, 0, 2))

    def decode(self, states, points_full):
        return self.decoder(states, points_full)

    def _encode_target_sets(self, target_features, condition):
        batch = tf.shape(target_features)[0]
        num_targets = tf.shape(target_features)[1]
        num_points = tf.shape(target_features)[2]
        feature_dim = tf.shape(target_features)[3]
        flat_features = tf.reshape(
            target_features, (batch * num_targets, num_points, feature_dim)
        )
        if condition is None:
            flat_condition = None
        else:
            flat_condition = tf.reshape(
                tf.broadcast_to(
                    tf.expand_dims(condition, axis=1),
                    (batch, num_targets, tf.shape(condition)[-1]),
                ),
                (batch * num_targets, tf.shape(condition)[-1]),
            )
        flat_latent, flat_embedding = self.target_encoder(flat_features, flat_condition)
        latent = tf.reshape(flat_latent, (batch, num_targets, self.target_encoder.latent_dim))
        embedding = tf.reshape(
            flat_embedding, (batch, num_targets, self.target_encoder.embedding_dim)
        )
        return latent, embedding

    def encode_targets(self, target_features, condition):
        _, embedding = self._encode_target_sets(target_features, condition)
        return embedding

    def encode_teacher_contexts(self, teacher_context_features, condition):
        return self._encode_target_sets(teacher_context_features, condition)

    def predict_targets(self, states_at_targets, target_times, condition, context_embedding):
        return self.predictor(states_at_targets, target_times, condition, context_embedding)

    def sync_target_encoder(self):
        for source, target in zip(
            self.context_encoder.variables, self.target_encoder.variables
        ):
            target.assign(source)

    def update_target_encoder(self, decay):
        decay = tf.cast(decay, self.context_encoder.variables[0].dtype)
        for source, target in zip(
            self.context_encoder.variables, self.target_encoder.variables
        ):
            target.assign(decay * target + (1.0 - decay) * source)
