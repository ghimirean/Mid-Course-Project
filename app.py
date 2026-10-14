"""Streamlit demo for the Sign Language MNIST CNN."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sign_mnist_cnn.keras"
# Sign Language MNIST uses labels 0-24 and omits J (9); Z is a motion sign.
CLASS_IDS = [label for label in range(25) if label != 9]
LABELS = [chr(ord("A") + label) for label in CLASS_IDS]


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_reference_examples():
    csv_path = ROOT / "data" / "sign_mnist_test.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def prepare_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """Convert an upload to the 28x28 grayscale input expected by the model."""
    gray = ImageOps.grayscale(image)
    # Fit rather than stretch so the uploaded sign keeps its proportions.
    gray.thumbnail((28, 28), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(gray, ((28 - gray.width) // 2, (28 - gray.height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32)
    # Sign MNIST foreground is bright on a dark background. Invert light-paper
    # photos and scans so their background matches the training images.
    if image.size != (28, 28) and np.mean(pixels) > 127:
        pixels = 255 - pixels
    return pixels.reshape(1, 28, 28, 1) / 255.0, canvas


def read_uploaded_image(upload) -> Image.Image:
    """Read common image formats plus the tiny ASCII PPM sample bundled here."""
    if upload.name.lower().endswith(".ppm"):
        tokens = []
        for line in upload.getvalue().decode("ascii").splitlines():
            tokens.extend(line.split("#", 1)[0].split())
        if len(tokens) < 4 or tokens[0] != "P3":
            raise ValueError("Only ASCII P3 PPM files are supported for .ppm uploads.")
        width, height, max_value = map(int, tokens[1:4])
        values = np.asarray(tokens[4:], dtype=np.float32)
        if max_value <= 0 or values.size != width * height * 3:
            raise ValueError("The PPM image has an invalid header or pixel count.")
        pixels = np.clip(values * (255.0 / max_value), 0, 255).astype(np.uint8)
        return Image.fromarray(pixels.reshape(height, width, 3), mode="RGB")
    return Image.open(upload).convert("RGB")


st.set_page_config(page_title="Sign Language Lab", page_icon="🤟", layout="wide")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#14243b; --muted:#65748a; --teal:#087f8c; --line:#e3e9f0; }
      .stApp { background: #f4f7fb; color:var(--ink); }
      [data-testid="stHeader"] { background:transparent; }
      .block-container { max-width:1120px; padding:2.2rem 2rem 4rem; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      h1,h2,h3 { font-family:'Manrope',sans-serif; color:var(--ink); letter-spacing:-.035em; }
      .hero { position:relative; overflow:hidden; padding:2.15rem 2.4rem; margin-bottom:1.6rem;
        border-radius:24px; color:#fff; background:linear-gradient(115deg,#102f49 0%,#075c69 64%,#13858c 100%);
        box-shadow:0 18px 44px rgba(20,53,75,.16); }
      .hero:after { content:'🤟'; position:absolute; right:4%; top:-1.35rem; font-size:10rem; opacity:.12; transform:rotate(-12deg); }
      .hero-kicker { text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:700; color:#a9e4df; }
      .hero h1 { color:white; font-size:clamp(2rem,4vw,3rem); margin:.45rem 0 .35rem; }
      .hero p { color:#d8e8ed; max-width:650px; margin:0; font-size:1.02rem; }
      .eyebrow { color:var(--teal); font-size:.76rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
      .stTabs [data-baseweb="tab-list"] { gap:.6rem; border-bottom:1px solid var(--line); }
      .stTabs [data-baseweb="tab"] { height:3.2rem; border-radius:12px 12px 0 0; padding:0 1.25rem; color:#526176; font-weight:600; }
      .stTabs [aria-selected="true"] { color:#087f8c !important; background:#e8f5f4; }
      .stTabs [data-baseweb="tab-highlight"] { background:#087f8c; height:3px; }
      [data-testid="stFileUploader"] section { background:white; border:1.5px dashed #aebdca; border-radius:16px; }
      [data-testid="stFileUploader"] section:hover { border-color:#087f8c; }
      [data-testid="stMetric"] { background:white; padding:1.2rem 1.35rem; border:1px solid var(--line); border-radius:16px; box-shadow:0 5px 18px rgba(27,49,70,.045); }
      [data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
      [data-testid="stMetricValue"] { color:#087f8c; font-family:'Manrope',sans-serif; }
      [data-testid="stAlert"] { border-radius:14px; }
      [data-testid="stImage"] img { border-radius:16px; }
      div[data-testid="stCaptionContainer"] { color:var(--muted); }
      .helper { background:white; border:1px solid var(--line); border-radius:16px; padding:1rem 1.2rem; color:#53647a; }
      .helper strong { color:var(--ink); }
      .stButton button { border-radius:10px; }
      @media (max-width:700px) { .block-container { padding:1.2rem 1rem 3rem; } .hero { padding:1.6rem; } .hero:after { right:-1rem; font-size:7rem; } }
    </style>
    <div class="hero">
      <div class="hero-kicker">Interactive machine learning demo</div>
      <h1>Sign Language Lab</h1>
      <p>Explore how a neural network reads static ASL handshapes, one letter at a time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="helper"><strong>About this model</strong> &nbsp; Trained on centered 28 × 28 grayscale Sign Language MNIST images. '
    'J and Z are excluded; arbitrary photos may be less reliable than dataset-style images.</div>',
    unsafe_allow_html=True,
)
st.write("")

model = get_model()
if model is None:
    st.warning("The trained model file is not in this checkout yet.")
    st.code("python -m pip install -r requirements.txt\npython src/train_model.py\nstreamlit run app.py", language="bash")
    st.caption("Place sign_mnist_train.csv and sign_mnist_test.csv in data/ before training.")
else:
    sample_tab, upload_tab = st.tabs(["Check on a dataset example", "Try your own image"])
    with sample_tab:
        st.markdown('<div class="eyebrow">Model check</div>', unsafe_allow_html=True)
        st.subheader("Test with a known letter")
        st.write("Choose a letter and inspect a held-out example with its expected label.")
        examples = load_reference_examples()
        if examples is None:
            st.caption("For this check, place sign_mnist_test.csv in data/. This file is part of the Kaggle dataset.")
        else:
            target_letter = st.selectbox("Choose a letter to test", LABELS, index=LABELS.index("M"))
            target_label = CLASS_IDS[LABELS.index(target_letter)]
            matching_rows = examples[examples["label"] == target_label]
            sample_number = st.slider(
                f"{target_letter} example", min_value=1, max_value=len(matching_rows), value=1
            )
            row = matching_rows.iloc[sample_number - 1]
            sample_image = Image.fromarray(
                row.drop(labels="label").to_numpy(dtype=np.uint8).reshape(28, 28), mode="L"
            )
            sample = np.asarray(sample_image, dtype=np.float32)
            sample = (sample / 255.0).reshape(1, 28, 28, 1)
            scores = np.asarray(model.predict(sample, verbose=0))[0]
            truth = int(row["label"])
            winner = int(np.argmax(scores))
            c1, c2, c3 = st.columns([1.1, 1, 1])
            c1.image(sample_image.resize((224, 224)), caption="Held-out sign image", width=224)
            c2.metric("Model prediction", "J (unused)" if winner == 9 else chr(ord("A") + winner),
                      f"{scores[winner] * 100:.1f}% confidence")
            expected = "J (unused)" if truth == 9 else chr(ord("A") + truth)
            c3.metric("Expected letter", expected, "Correct match" if winner == truth else "Different from expected")
            st.caption("This example comes from the held-out Kaggle test set; it was not used to fit the model.")

    with upload_tab:
        st.markdown('<div class="eyebrow">Your turn</div>', unsafe_allow_html=True)
        st.subheader("Upload a handshape")
        st.write("Use one clear, tightly cropped static letter sign. A plain background and centered hand give the model its best chance.")
        upload = st.file_uploader("Choose a hand-sign image", type=["png", "jpg", "jpeg", "bmp", "ppm"])
        if upload:
            try:
                image = read_uploaded_image(upload)
                model_input, preview = prepare_image(image)
                left, right = st.columns([1, 1])
                left.image(image, caption="Your image", use_container_width=True)
                right.image(preview.resize((224, 224)), caption="What the model sees (28 × 28)", width=224)
                scores = np.asarray(model.predict(model_input, verbose=0))[0]
                if len(scores) != 25:
                    st.error(f"Expected 25 output scores, but model returned {len(scores)}.")
                else:
                    valid_scores = scores[CLASS_IDS]
                    order = np.argsort(valid_scores)[::-1][:3]
                    best = int(order[0])
                    st.markdown('<div class="eyebrow">Prediction</div>', unsafe_allow_html=True)
                    result, alternatives = st.columns([1, 1.2])
                    result.metric("Most likely letter", LABELS[best], f"{valid_scores[best] * 100:.1f}% model score")
                    with alternatives:
                        st.markdown("**Other possible letters**")
                        for idx in order[1:]:
                            st.write(f"{LABELS[int(idx)]} &nbsp; · &nbsp; {valid_scores[idx] * 100:.1f}%", unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Could not read or classify that image: {exc}")
