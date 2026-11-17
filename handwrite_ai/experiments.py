"""Real implementations of the ten prescribed Deep Learning experiments.

All deep-learning experiments require TensorFlow; the Streamlit dashboard, logic
module and scikit-learn demonstration can start without TensorFlow.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from .config import ARTIFACTS, SEED, ensure_dirs
from .datasets import load_mnist, load_pendigits, sliding_windows
from .models import mlp, lenet, autoencoder, vgg16_transfer, stroke_lstm, gan_models
from .logic import gate
from .store import save_experiment
from .charts import digit_grid, line_plot


def _seed():
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)


def _compile(model, optimizer="adam"):
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _fit_classifier(model, data, epochs: int, batch_size: int = 128, shuffle=True):
    hist = model.fit(data.train_x, data.train_y,
                     validation_data=(data.val_x, data.val_y),
                     epochs=epochs, batch_size=batch_size, shuffle=shuffle, verbose=2)
    loss, acc = model.evaluate(data.test_x, data.test_y, verbose=0)
    preds = np.argmax(model.predict(data.test_x, verbose=0), axis=1)
    return hist, {"test_loss": float(loss), "test_accuracy": float(acc),
                  "test_examples": int(len(data.test_y)),
                  "training_examples": int(len(data.train_y))}, preds


def _save_confusion(y_true, pred, name: str):
    cm = confusion_matrix(y_true, pred, labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=np.arange(10)).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"{name}: held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / f"{name}_confusion.png", dpi=140)
    plt.close(fig)


def run_logic():
    rows = [{"a": a, "b": b, "AND": gate("AND", a, b),
             "OR": gate("OR", a, b), "NAND": gate("NAND", a, b)}
            for a in (0, 1) for b in (0, 1)]
    save_experiment("logic", {"truth_table": rows,
                              "not_table": [{"a": a, "NOT": gate("NOT", a)}
                                            for a in (0, 1)]})
    print(json.dumps(rows, indent=2))


def run_perceptron(train_limit=12000, test_limit=2000):
    d = load_mnist(train_limit, test_limit)
    train_mask = np.isin(d.train_y, [0, 1])
    test_mask = np.isin(d.test_y, [0, 1])
    xtrain = d.train_x[train_mask].reshape(-1, 784)
    ytrain = d.train_y[train_mask]
    xtest = d.test_x[test_mask].reshape(-1, 784)
    ytest = d.test_y[test_mask]
    model = Perceptron(max_iter=1000, tol=1e-4, random_state=SEED)
    model.fit(xtrain, ytrain)
    preds = model.predict(xtest)
    acc = float(accuracy_score(ytest, preds))
    np.savez_compressed(ARTIFACTS / "perceptron.npz",
                        coef=model.coef_, intercept=model.intercept_, classes=model.classes_)
    save_experiment("perceptron", {"test_accuracy": acc, "classes": [0, 1],
                                    "test_examples": len(ytest),
                                    "training_examples": len(ytrain)})
    print(f"Perceptron (0 vs 1) test accuracy: {acc:.4f}")


def run_classifier(name: str, train_limit=12000, test_limit=2000,
                   epochs=5, batch_size=128):
    _seed()
    d = load_mnist(train_limit, test_limit)
    model = mlp() if name == "mlp" else lenet()
    _compile(model)
    hist, metrics, preds = _fit_classifier(model, d, epochs, batch_size)
    model.save(ARTIFACTS / f"{name}.keras")
    _save_confusion(d.test_y, preds, name)
    save_experiment(name, metrics, hist.history)
    print(f"{name} held-out test accuracy: {metrics['test_accuracy']:.4f}")


def run_optimizers(train_limit=12000, test_limit=2000,
                   epochs=5, batch_size=128):
    import tensorflow as tf
    _seed()
    d = load_mnist(train_limit, test_limit)
    shared_weights = mlp().get_weights()
    opts = {
        "SGD": lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
        "RMSProp": lambda: tf.keras.optimizers.RMSprop(learning_rate=0.001),
        "Adam": lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
    }
    outcomes, histories = {}, {}
    for name, opt_factory in opts.items():
        _seed()
        model = mlp()
        model.set_weights(shared_weights)
        _compile(model, opt_factory())
        hist, metrics, _ = _fit_classifier(model, d, epochs, batch_size, shuffle=False)
        outcomes[name] = metrics
        histories[name] = hist.history
    save_experiment("optimizers", {"models": outcomes,
                                  "learning_rates": {"SGD": 0.01, "RMSProp": 0.001,
                                                     "Adam": 0.001},
                                  "note": "Same initial weights and shuffle=False for reproducibility"}, histories)
    for metric in ("val_accuracy", "val_loss"):
        fig = line_plot({name: h[metric] for name, h in histories.items()},
                        f"Optimizer comparison: {metric}", metric)
        fig.savefig(ARTIFACTS / f"optimizers_{metric}.png", dpi=140)
        plt.close(fig)
    print("Optimizer test metrics:", outcomes)


def run_regularization(train_limit=12000, test_limit=2000,
                       epochs=8, batch_size=128):
    _seed()
    d = load_mnist(train_limit, test_limit)
    variants = {"Baseline": (False, False), "Dropout": (True, False),
                "BatchNorm": (False, True), "Both": (True, True)}
    outcomes, histories = {}, {}
    for name, (drop, bn) in variants.items():
        _seed()
        model = mlp(dropout=drop, batchnorm=bn)
        _compile(model)
        hist, metrics, _ = _fit_classifier(model, d, epochs, batch_size)
        tr_loss, tr_acc = model.evaluate(d.train_x, d.train_y, verbose=0)
        val_loss, val_acc = model.evaluate(d.val_x, d.val_y, verbose=0)
        metrics.update({"train_accuracy": float(tr_acc),
                        "validation_accuracy": float(val_acc),
                        "generalization_gap": float(tr_acc - val_acc),
                        "train_loss": float(tr_loss), "validation_loss": float(val_loss)})
        outcomes[name] = metrics
        histories[name] = hist.history
    save_experiment("regularization", {"models": outcomes}, histories)
    for metric in ("val_accuracy", "val_loss"):
        fig = line_plot({name: h[metric] for name, h in histories.items()},
                        f"Regularization comparison: {metric}", metric)
        fig.savefig(ARTIFACTS / f"regularization_{metric}.png", dpi=140)
        plt.close(fig)
    print("Regularization test metrics:", outcomes)


def run_compression(train_limit=12000, test_limit=2000,
                    epochs=8, batch_size=256, latent_dim=32):
    _seed()
    d = load_mnist(train_limit, test_limit)
    xtr = d.train_x.reshape(-1, 784)
    xv = d.val_x.reshape(-1, 784)
    xt = d.test_x.reshape(-1, 784)
    if latent_dim >= min(xtr.shape):
        raise ValueError("Latent dimension must be smaller than the dataset dimensions")
    pca = PCA(n_components=latent_dim, svd_solver="randomized", random_state=SEED)
    pca.fit(xtr)
    pca_recon = pca.inverse_transform(pca.transform(xt))
    model, encoder = autoencoder(latent_dim)
    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(xtr, xtr, validation_data=(xv, xv),
                     batch_size=batch_size, epochs=epochs, verbose=2)
    ae_recon = model.predict(xt, verbose=0)
    metrics = {"pca_mse": float(np.mean((xt - pca_recon)**2)),
               "autoencoder_mse": float(np.mean((xt - ae_recon)**2)),
               "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
               "original_dim": 784, "latent_dim": latent_dim,
               "test_examples": len(xt)}
    np.savez_compressed(ARTIFACTS / "pca.npz", mean=pca.mean_,
                        components=pca.components_, latent_dim=latent_dim)
    model.save(ARTIFACTS / "autoencoder.keras")
    encoder.save(ARTIFACTS / "encoder.keras")
    save_experiment("compression", metrics, hist.history)
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    for i in range(8):
        for r, imgs in enumerate((xt, pca_recon, ae_recon)):
            axes[r, i].imshow(imgs[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[r, i].axis("off")
        axes[0, i].set_title(str(d.test_y[i]))
    for row, title in enumerate(("Original", "PCA", "Autoencoder")):
        axes[row, 0].set_ylabel(title, rotation=90, labelpad=14, fontsize=10)
    fig.suptitle("MNIST reconstruction: original / PCA / Autoencoder")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "compression_comparison.png", dpi=140)
    plt.close(fig)
    print("Reconstruction MSE:", metrics)


def _image_directory(path: Path, image_size: int, batch_size: int,
                     class_names: list[str] | None = None, shuffle=False):
    import tensorflow as tf
    kw = {"directory": str(path), "image_size": (image_size, image_size),
          "batch_size": batch_size, "label_mode": "int", "shuffle": shuffle}
    if class_names is not None:
        kw["class_names"] = class_names
    return tf.keras.utils.image_dataset_from_directory(**kw)


def run_transfer(custom_data: str | Path, epochs=4, batch_size=16,
                 image_size=96):
    import tensorflow as tf
    _seed()
    root = Path(custom_data)
    folders = [root / k for k in ("train", "val", "test")]
    if not all(p.is_dir() for p in folders):
        raise ValueError("Custom dataset requires train/, val/, test/ directories, "
                         "each with identical class subdirectories (0 through 9).")
    raw_train = _image_directory(folders[0], image_size, batch_size, shuffle=True)
    classes = raw_train.class_names
    if len(classes) != 10:
        raise ValueError(f"Expected 10 digit classes, got {classes}")
    raw_val = _image_directory(folders[1], image_size, batch_size, classes)
    raw_test = _image_directory(folders[2], image_size, batch_size, classes)
    def transform(images, labels):
        return tf.keras.applications.vgg16.preprocess_input(images), labels
    autotune = tf.data.AUTOTUNE
    tr = raw_train.map(transform, num_parallel_calls=autotune).prefetch(autotune)
    va = raw_val.map(transform, num_parallel_calls=autotune).prefetch(autotune)
    te = raw_test.map(transform, num_parallel_calls=autotune).prefetch(autotune)
    model = vgg16_transfer(num_classes=len(classes), image_size=image_size)
    _compile(model, tf.keras.optimizers.Adam(learning_rate=1e-4))
    hist = model.fit(tr, validation_data=va, epochs=epochs, verbose=2)
    test_loss, test_acc = model.evaluate(te, verbose=0)
    predictions, truths = [], []
    for images, labels in te:
        predictions.extend(np.argmax(model.predict(images, verbose=0), axis=1).tolist())
        truths.extend(labels.numpy().tolist())
    model.save(ARTIFACTS / "vgg16.keras")
    (ARTIFACTS / "vgg16_classes.json").write_text(json.dumps(classes), encoding="utf-8")
    (ARTIFACTS / "vgg16_image_size.json").write_text(json.dumps(image_size), encoding="utf-8")
    metrics = {"test_accuracy": float(test_acc), "test_loss": float(test_loss),
               "test_examples": len(truths), "class_names": classes,
               "imagenet_pretraining": True, "frozen_backbone": True}
    save_experiment("vgg16", metrics, hist.history)
    _save_confusion(truths, predictions, "vgg16")
    print("VGG16 test metrics:", metrics)


def run_lstm(pendigits_dir=None, epochs=12, batch_size=128, window=3):
    _seed()
    train, test = load_pendigits(pendigits_dir)
    # Keep the original UCI train/test split; separate validation from train only.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train))
    val_size = max(1, int(len(train)*0.15))
    val_traj, train_traj = train[order[:val_size]], train[order[val_size:]]
    xtr, ytr = sliding_windows(train_traj, window)
    xv, yv = sliding_windows(val_traj, window)
    xt, yt = sliding_windows(test, window)
    model = stroke_lstm(window)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    hist = model.fit(xtr, ytr, epochs=epochs, batch_size=batch_size,
                     validation_data=(xv, yv), verbose=2)
    mse, mae = model.evaluate(xt, yt, verbose=0)
    model.save(ARTIFACTS / "stroke_lstm.keras")
    np.savez_compressed(ARTIFACTS / "stroke_samples.npz",
                        samples=test[:50], window=window)
    metrics = {"test_mse": float(mse), "test_mae": float(mae),
               "test_windows": len(xt), "window": window,
               "note": "PenDigits points are spatially resampled, not fixed-time timestamps"}
    save_experiment("lstm", metrics, hist.history)
    print("LSTM next-coordinate metrics:", metrics)


def run_gan(train_limit=12000, epochs=10, batch_size=128, noise_dim=100):
    import tensorflow as tf
    _seed()
    d = load_mnist(train_limit, test_limit=100)
    generator, discriminator = gan_models(noise_dim)
    gen_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    disc_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    dataset = (tf.data.Dataset.from_tensor_slices(d.train_x * 2.0 - 1.0)
               .shuffle(buffer_size=min(len(d.train_x), 6000), seed=SEED)
               .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    gen_loss_history, disc_loss_history = [], []
    for epoch in range(epochs):
        g_losses, d_losses = [], []
        for real in dataset:
            batch_n = tf.shape(real)[0]
            noise = tf.random.normal((batch_n, noise_dim))
            with tf.GradientTape() as gt, tf.GradientTape() as dt:
                fake = generator(noise, training=True)
                real_logits = discriminator(real, training=True)
                fake_logits = discriminator(fake, training=True)
                gl = bce(tf.ones_like(fake_logits), fake_logits)
                dl = (bce(tf.ones_like(real_logits), real_logits) +
                      bce(tf.zeros_like(fake_logits), fake_logits))
            g_grads = gt.gradient(gl, generator.trainable_variables)
            d_grads = dt.gradient(dl, discriminator.trainable_variables)
            gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
            disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
            g_losses.append(float(gl.numpy()))
            d_losses.append(float(dl.numpy()))
        gen_loss_history.append(float(np.mean(g_losses)))
        disc_loss_history.append(float(np.mean(d_losses)))
        print(f"Epoch {epoch+1}/{epochs}: G={gen_loss_history[-1]:.4f}, "
              f"D={disc_loss_history[-1]:.4f}")
    generator.save(ARTIFACTS / "generator.keras")
    noise = tf.random.normal((16, noise_dim))
    images = (generator.predict(noise, verbose=0) + 1.0) / 2.0
    fig = digit_grid(images, "GAN-generated samples (unconditional; no class selection)")
    fig.savefig(ARTIFACTS / "gan_samples.png", dpi=140)
    plt.close(fig)
    save_experiment("gan", {"epochs": epochs, "trained_images": len(d.train_x),
                            "noise_dim": noise_dim, "conditional": False},
                    {"generator_loss": gen_loss_history,
                     "discriminator_loss": disc_loss_history})
    print("Saved trained unconditional GAN generator and generated samples")
