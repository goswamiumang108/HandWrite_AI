"""HandWrite AI — interactive capstone dashboard.

Run: streamlit run app.py
All saved metrics shown here come from actual training, not example numbers.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from handwrite_ai.config import ARTIFACTS, ASSETS, ensure_dirs
from handwrite_ai.store import load_experiment, saved_experiments
from handwrite_ai.logic import gate
from handwrite_ai.preprocessing import prepare_digit
from handwrite_ai.demo import train_demo, demo_predict
from handwrite_ai.charts import line_plot, digit_grid

st.set_page_config(
    page_title="HandWrite AI | Deep Learning Capstone",
    page_icon="✍",
    layout="wide",
    initial_sidebar_state="expanded",
)
ensure_dirs()

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px}
.hero {padding: 1.4rem 1.7rem; border: 1px solid #334155; border-radius: 18px;
       background: linear-gradient(115deg, #14243b, #172b35); color: #f8fafc;}
.hero h1 {font-size: 2.45rem; margin: 0 0 .5rem 0}
.hero p {color: #d0e3f1; margin: 0}
.eyebrow {font-size: .77rem; letter-spacing: 2px; font-weight: 700; color: #8bd8e3;}
.small-note {font-size: .85rem; color: #64748b;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the offline recognition baseline...")
def cached_demo():
    return train_demo()


@st.cache_resource(show_spinner="Loading trained TensorFlow model...")
def load_keras_model(path_string: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path_string, compile=False)


def image_from_ui(key: str):
    method = st.radio("Input method", ["Draw", "Upload", "Sample"],
                      horizontal=True, key=f"{key}_method")
    if method == "Draw":
        from handwrite_ai.drawpad import draw_digit
        image = draw_digit(key=f"{key}_drawpad")
        return image, "Hand-drawn input"
    if method == "Upload":
        upload = st.file_uploader("Upload a handwritten digit image",
                                  type=["png", "jpg", "jpeg", "webp"], key=key)
        if upload is None:
            return None, ""
        try:
            return Image.open(upload).convert("RGB"), "Uploaded image"
        except Exception:
            st.error("Could not read the uploaded image")
            return None, ""
    selection = st.selectbox("Choose a built-in example", [f"Example {i+1}" for i in range(12)],
                             key=f"{key}_sample")
    _, _, imgs, labels = cached_demo()
    idx = int(selection.split()[-1]) - 1
    img = Image.fromarray((imgs[idx] / 16 * 255).astype("uint8"), "L")
    return img.resize((240, 240), Image.Resampling.NEAREST), f"Sample (true digit: {labels[idx]})"


def try_load(name: str):
    f = ARTIFACTS / f"{name}.keras"
    if not f.exists():
        st.warning(f"The {name} model is not trained. Run `python -m scripts.train {name}` first.")
        return None
    try:
        return load_keras_model(str(f))
    except (ImportError, ModuleNotFoundError) as exc:
        st.error(f"TensorFlow is required to load this model: {exc}")
    except Exception as exc:
        st.error(f"Unable to load {f.name}: {exc}")
    return None


def show_prediction(probs: np.ndarray, classes=None):
    p = np.asarray(probs).reshape(-1)
    classes = classes or [str(i) for i in range(len(p))]
    order = np.argsort(p)[::-1]
    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted class", classes[int(order[0])])
        st.metric("Model confidence", f"{100*p[order[0]]:.1f}%")
    with right:
        st.bar_chart(pd.DataFrame({"Probability": p}, index=classes))
    st.caption("Softmax probabilities are model confidence scores, not guaranteed calibrated probabilities.")


def history_comparison(name: str, record: dict):
    st.write("**Held-out test metrics**")
    models = record.get("metrics", {}).get("models", {})
    if models:
        st.dataframe(pd.DataFrame.from_dict(models, orient="index"), use_container_width=True)
    histories = record.get("history", {})
    if not histories:
        return
    metric = st.selectbox("Training history", ["val_accuracy", "val_loss", "accuracy", "loss"],
                          key=f"metric_{name}")
    entries = {model: h.get(metric, []) for model, h in histories.items()}
    fig = line_plot(entries, f"{name.title()} — {metric}", metric)
    st.pyplot(fig)
    plt.close(fig)


def show_coverage():
    checks = [
        ("01", "McCulloch–Pitts gates", "logic"),
        ("02", "Perceptron 0/1", "perceptron"),
        ("03", "MLP backpropagation", "mlp"),
        ("04", "SGD / RMSProp / Adam", "optimizers"),
        ("05", "PCA vs Autoencoder", "compression"),
        ("06", "Dropout + BatchNorm", "regularization"),
        ("07", "LeNet-5 classification", "lenet"),
        ("08", "VGG16 transfer learning", "vgg16"),
        ("09", "LSTM stroke prediction", "lstm"),
        ("10", "GAN digit generation", "gan"),
    ]
    ready = saved_experiments()
    df = pd.DataFrame([{"Practical": n, "Module": label,
                        "Status": "Completed" if artifact in ready else "Not trained"}
                       for n, label, artifact in checks])
    st.dataframe(df, use_container_width=True, hide_index=True)
    return len(ready.keys() & {c[2] for c in checks})


st.sidebar.markdown("### HandWrite AI")
page = st.sidebar.radio("Navigate", [
    "Overview", "Recognize digits", "Training comparisons", "Compress & reconstruct",
    "Generate digits", "Stroke prediction", "Neuron fundamentals", "Training guide"
])
st.sidebar.divider()
st.sidebar.caption("Deep Learning Foundations | 10 practicals | Final-year capstone")
st.sidebar.caption("Predictions and test metrics are genuine model outputs. Missing models are never simulated.")

if page == "Overview":
    st.markdown("""
<div class="hero"><div class="eyebrow">DEEP LEARNING FOUNDATIONS · CAPSTONE</div>
<h1>HandWrite AI</h1>
<p>An Intelligent Handwriting Recognition, Analysis and Generation System</p></div>
""", unsafe_allow_html=True)
    st.write("")
    completed = len(saved_experiments())
    score = cached_demo()[1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Practical modules", "10")
    c2.metric("Saved experiment reports", str(completed))
    c3.metric("Offline 8×8 demo test accuracy", f"{score:.1%}")
    st.caption("Offline demo uses scikit-learn's separate 8×8 digit dataset, not MNIST. Its measured score must not be reported as CNN accuracy.")
    arch = ASSETS / "architecture.png"
    if arch.exists():
        st.image(str(arch), caption="Integrated system modules", use_container_width=True)
    st.subheader("Practical coverage")
    show_coverage()

elif page == "Recognize digits":
    st.title("Handwritten Digit Recognition")
    st.write("Use the immediately available 8×8 baseline, or select an MNIST model after training.")
    choices = ["Offline demo (scikit-learn)", "MLP (MNIST)", "LeNet-5 (MNIST)",
               "VGG16 (custom dataset)", "Perceptron (0 vs 1 only)"]
    choice = st.selectbox("Classification model", choices)
    image, caption = image_from_ui("recognition")
    if image is not None:
        col, col2 = st.columns([1, 2])
        with col:
            st.image(image, caption=caption, width=220)
        if st.button("Recognize digit", type="primary"):
            try:
                if choice.startswith("Offline"):
                    arr = prepare_digit(image, target_size=8, digit_size=6)
                    probs = demo_predict(arr)
                    st.caption("Offline baseline: 8×8 images, logistic regression; uploaded handwriting may differ from training images.")
                    st.image(arr, caption="Processed 8×8 input", width=120, clamp=True)
                    show_prediction(probs)
                elif choice.startswith("Perceptron"):
                    path = ARTIFACTS / "perceptron.npz"
                    if not path.exists():
                        st.warning("Train first: `python -m scripts.train perceptron`")
                    else:
                        arr = prepare_digit(image).reshape(1, -1)
                        with np.load(path, allow_pickle=False) as p:
                            score = float(arr @ p["coef"].T + p["intercept"])
                            pred = int(p["classes"][int(score > 0)])
                        st.metric("Predicted binary class", str(pred))
                        st.caption("Trained only on digits 0 and 1; do not use for other digits.")
                else:
                    name = "mlp" if choice.startswith("MLP") else ("lenet" if choice.startswith("LeNet") else "vgg16")
                    model = try_load(name)
                    if model is not None:
                        arr = prepare_digit(image)
                        if name == "vgg16":
                            import tensorflow as tf
                            sizepath = ARTIFACTS / "vgg16_image_size.json"
                            size = json.loads(sizepath.read_text()) if sizepath.exists() else 96
                            triplet = np.stack([arr * 255.0]*3, axis=-1)
                            x = tf.image.resize(triplet, (size, size)).numpy()[None]
                            x = tf.keras.applications.vgg16.preprocess_input(x)
                            clsfile = ARTIFACTS / "vgg16_classes.json"
                            classes = json.loads(clsfile.read_text()) if clsfile.exists() else None
                        else:
                            x = arr[None, ..., None]
                            classes = None
                        probs = model.predict(x, verbose=0)[0]
                        st.image(arr, caption="Processed 28×28 input", width=140, clamp=True)
                        show_prediction(probs, classes)
            except ValueError as exc:
                st.error(str(exc))
    else:
        st.info("Draw, upload, or select a digit to classify it.")

elif page == "Training comparisons":
    st.title("Model Performance Laboratory")
    topic = st.selectbox("Choose comparison", ["Optimizers", "Regularization", "MLP vs LeNet-5", "VGG16"])
    if topic in ("Optimizers", "Regularization"):
        key = "optimizers" if topic == "Optimizers" else "regularization"
        record = load_experiment(key)
        if record:
            history_comparison(key, record)
            if key == "optimizers":
                st.caption("Comparison uses shared initialization and fixed training configuration. Different learning rates are intentional.")
        else:
            st.info(f"Train this analysis: `python -m scripts.train {key}`")
    elif topic == "MLP vs LeNet-5":
        records = {n: load_experiment(n) for n in ("mlp", "lenet")}
        available = {n: r["metrics"] for n, r in records.items() if r}
        if available:
            st.dataframe(pd.DataFrame.from_dict(available, orient="index"), use_container_width=True)
            hist = {n: r["history"].get("val_accuracy", []) for n, r in records.items() if r}
            fig = line_plot(hist, "Validation accuracy", "Accuracy")
            st.pyplot(fig)
            plt.close(fig)
            for n in available:
                image_file = ARTIFACTS / f"{n}_confusion.png"
                if image_file.exists():
                    st.image(str(image_file), caption=f"{n.upper()} test confusion matrix")
        else:
            st.info("Train `mlp` and `lenet` experiments first.")
    else:
        record = load_experiment("vgg16")
        if record:
            st.json(record["metrics"])
            image_file = ARTIFACTS / "vgg16_confusion.png"
            if image_file.exists():
                st.image(str(image_file))
        else:
            st.info("Prepare a custom dataset and run `python -m scripts.train transfer --custom-data data/custom_digits`.")

elif page == "Compress & reconstruct":
    st.title("Image Compression: PCA vs Autoencoder")
    rec = load_experiment("compression")
    if rec:
        st.json(rec["metrics"])
        image, caption = image_from_ui("compression")
        if image is not None:
            if st.button("Reconstruct image", type="primary"):
                model = try_load("autoencoder")
                pca_file = ARTIFACTS / "pca.npz"
                if model is not None and pca_file.exists():
                    try:
                        original = prepare_digit(image).reshape(1, 784)
                        with np.load(pca_file, allow_pickle=False) as p:
                            reduced = (original - p["mean"]) @ p["components"].T
                            pca_image = (reduced @ p["components"] + p["mean"]).reshape(28, 28)
                        ae_image = model.predict(original, verbose=0)[0].reshape(28, 28)
                        cols = st.columns(3)
                        for col, title, im in zip(cols,
                                ("Original", "PCA reconstruction", "Autoencoder reconstruction"),
                                (original.reshape(28, 28), pca_image, ae_image)):
                            col.image(np.clip(im, 0, 1), caption=title, width=200, clamp=True)
                        st.write({"PCA image MSE": float(np.mean((original.reshape(28,28)-pca_image)**2)),
                                  "Autoencoder image MSE": float(np.mean((original.reshape(28,28)-ae_image)**2))})
                    except ValueError as exc:
                        st.error(str(exc))
        path = ARTIFACTS / "compression_comparison.png"
        if path.exists():
            st.image(str(path), caption="Held-out MNIST test examples")
    else:
        st.info("Train first: `python -m scripts.train compression`. Then compare reconstructions here.")

elif page == "Generate digits":
    st.title("Synthetic Handwriting Generator")
    st.write("A basic **unconditional GAN** learns the MNIST distribution. It cannot guarantee a selected digit class.")
    rec = load_experiment("gan")
    if rec:
        st.json(rec["metrics"])
        sample = ARTIFACTS / "gan_samples.png"
        if sample.exists():
            st.image(str(sample), caption="Samples saved at the end of GAN training")
        if st.button("Generate 16 new images", type="primary"):
            model = try_load("generator")
            if model is not None:
                import tensorflow as tf
                noise_dim = int(rec["metrics"]["noise_dim"])
                random_noise = tf.random.normal((16, noise_dim))
                imgs = (model.predict(random_noise, verbose=0) + 1) / 2
                fig = digit_grid(imgs, "New samples from trained unconditional GAN")
                st.pyplot(fig)
                plt.close(fig)
        history = rec.get("history", {})
        if history:
            fig = line_plot(history, "Adversarial training losses", "Binary crossentropy")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Train first: `python -m scripts.train gan`. GAN training requires more epochs for visually useful results.")

elif page == "Stroke prediction":
    st.title("LSTM: Predict the Next Pen Coordinate")
    st.write("UCI PenDigits provides eight ordered, spatially resampled coordinate pairs per handwritten digit. They are **not** real-time timestamps.")
    model = None
    samplesfile = ARTIFACTS / "stroke_samples.npz"
    rec = load_experiment("lstm")
    if rec and samplesfile.exists():
        st.json(rec["metrics"])
        with np.load(samplesfile, allow_pickle=False) as data:
            samples, window = data["samples"].copy(), int(data["window"])
        idx = st.slider("Example stroke", 0, len(samples)-1, 0)
        start = st.slider("Prediction step", window, len(samples[idx])-1, window)
        if st.button("Predict next coordinate", type="primary"):
            model = try_load("stroke_lstm")
            if model is not None:
                trajectory = samples[idx]
                x = trajectory[start-window:start][None].astype("float32")
                pred = model.predict(x, verbose=0)[0]
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(trajectory[:, 0], trajectory[:, 1], "o-", label="Actual stroke")
                ax.scatter(trajectory[start, 0], trajectory[start, 1], marker="x", s=120,
                           label="Next actual point")
                ax.scatter(pred[0], pred[1], marker="*", s=140,
                           label="LSTM prediction")
                ax.set(xlim=(-0.05, 1.05), ylim=(1.05, -0.05),
                       xlabel="Normalized X", ylabel="Normalized Y")
                ax.legend()
                ax.set_aspect("equal")
                st.pyplot(fig)
                plt.close(fig)
                st.write({"Actual XY": trajectory[start].round(3).tolist(),
                          "Predicted XY": pred.round(3).tolist()})
    else:
        st.info("Train first: `python -m scripts.train lstm`. PenDigits may need downloading on the first run.")

elif page == "Neuron fundamentals":
    st.title("Neuron Fundamentals")
    st.subheader("McCulloch–Pitts logic gate simulator")
    c1, c2, c3 = st.columns(3)
    selected = c1.selectbox("Gate", ["AND", "OR", "NAND", "NOT"])
    a = c2.selectbox("A", [0, 1])
    b = c3.selectbox("B", [0, 1], disabled=(selected == "NOT"))
    st.metric("Neuron output", gate(selected, a, b))
    st.caption("A threshold neuron implements the specified Boolean function; NOT uses inhibitory weight −1.")
    st.subheader("Binary perceptron: digit 0 vs digit 1")
    rec = load_experiment("perceptron")
    if rec:
        st.json(rec["metrics"])
    else:
        st.info("Train the binary classifier with `python -m scripts.train perceptron`.")

elif page == "Training guide":
    st.title("Training and Reproducibility")
    st.write("Run the commands below in a terminal from the extracted project directory.")
    st.code("python -m pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    st.subheader("Run a short smoke test")
    st.code("python -m scripts.train logic\npython -m scripts.train perceptron --quick\npython -m scripts.train lenet --quick", language="bash")
    st.subheader("Train all MNIST-based experiments")
    st.code("python -m scripts.train mlp\npython -m scripts.train optimizers\npython -m scripts.train compression\npython -m scripts.train regularization\npython -m scripts.train lenet\npython -m scripts.train gan", language="bash")
    st.subheader("Train additional dataset experiments")
    st.code("python -m scripts.train lstm\npython -m scripts.train transfer --custom-data data/custom_digits", language="bash")
    st.write("VGG16 requires labeled images under `train/`, `val/`, `test/` with classes `0`–`9`. The ImageNet backbone downloads once.")
    st.write("For final reports, train with `--train-limit 0 --test-limit 0` and record hardware, epochs, dataset splits and random seed.")
    st.subheader("Experiment status")
    show_coverage()
