"""Model constructors (TensorFlow imported lazily so the demo runs without it)."""


def mlp(dropout: bool = False, batchnorm: bool = False):
    import tensorflow as tf
    inp = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inp)
    for units, rate in ((256, 0.35), (128, 0.25)):
        x = tf.keras.layers.Dense(units, use_bias=not batchnorm)(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        if dropout:
            x = tf.keras.layers.Dropout(rate)(x)
    out = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="mlp")


def lenet():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.ZeroPadding2D(2),
        tf.keras.layers.Conv2D(6, 5, activation="tanh"),
        tf.keras.layers.AveragePooling2D(2),
        tf.keras.layers.Conv2D(16, 5, activation="tanh"),
        tf.keras.layers.AveragePooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation="tanh"),
        tf.keras.layers.Dense(84, activation="tanh"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="lenet5_inspired")


def autoencoder(latent_dim: int = 32):
    import tensorflow as tf
    inp = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(256, activation="relu")(inp)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    z = tf.keras.layers.Dense(latent_dim, activation="linear", name="latent")(x)
    y = tf.keras.layers.Dense(64, activation="relu")(z)
    y = tf.keras.layers.Dense(256, activation="relu")(y)
    out = tf.keras.layers.Dense(784, activation="sigmoid")(y)
    model = tf.keras.Model(inp, out, name="autoencoder")
    encoder = tf.keras.Model(inp, z, name="encoder")
    return model, encoder


def stroke_lstm(length: int = 3):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.Input(shape=(length, 2)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2, activation="sigmoid"),
    ], name="stroke_lstm")


def vgg16_transfer(num_classes: int = 10, image_size: int = 96):
    import tensorflow as tf
    base = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet",
        input_shape=(image_size, image_size, 3), pooling="avg"
    )
    base.trainable = False
    inp = tf.keras.Input(shape=(image_size, image_size, 3))
    features = base(inp, training=False)
    x = tf.keras.layers.Dense(128, activation="relu")(features)
    x = tf.keras.layers.Dropout(0.3)(x)
    preds = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, preds, name="vgg16_transfer")


def gan_models(noise_dim: int = 100):
    import tensorflow as tf
    gen = tf.keras.Sequential([
        tf.keras.Input(shape=(noise_dim,)),
        tf.keras.layers.Dense(7 * 7 * 128, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        tf.keras.layers.Reshape((7, 7, 128)),
        tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding="same", activation="tanh"),
    ], name="generator")
    disc = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 4, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(64, 4, strides=2, padding="same"),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),  # logits
    ], name="discriminator")
    return gen, disc
