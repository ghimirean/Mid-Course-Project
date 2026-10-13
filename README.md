# Sign Language MNIST: deployment prototype

This repository includes a Streamlit prototype for classifying a cropped ASL hand-sign image with the project's three-hidden-layer MLP. The model predicts one of the 24 static Sign Language MNIST letters. J and Z are not in the dataset; J's label slot remains unused to match the notebook's 25-output model.

## Run locally

Use Python 3.10–3.12, then install the app dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The Kaggle archive can be downloaded and extracted with:

```bash
mkdir -p data
curl -fL https://www.kaggle.com/api/v1/datasets/download/datamunge/sign-language-mnist -o /tmp/sign-language-mnist.zip
unzip -o /tmp/sign-language-mnist.zip -d data
```

Then train and launch:

```bash
python src/train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit and upload a PNG, JPG, BMP, or PPM image. The trainer saves `models/sign_mnist_mlp.keras`; dataset and model files are excluded from Git by default.

## Deploy on Streamlit Community Cloud

1. Train the model locally with the dataset and confirm the prototype works.
2. Make the trained `models/sign_mnist_mlp.keras` artifact available in the GitHub repository used for deployment. Model artifacts are often too large for ordinary Git; if this one exceeds GitHub's file limit, store it in Git LFS or an artifact host and download it during app startup.
3. Push the app code, `requirements.txt`, and model artifact (or its download setup) to GitHub.
4. In Streamlit Community Cloud, create an app from that repository, set the main file to `app.py`, and deploy.

This checkout does not contain the CSV dataset, trained model artifact, or a configured hosting account, so it is ready for deployment but cannot produce a live hosted URL until those are supplied and connected.

## Input guidance

Use one hand centered in the frame against a plain dark background, with the palm facing the camera. The prototype converts uploads to grayscale, fits them to 28 × 28, and normalizes pixels to 0–1. Predictions on ordinary photographs may be less reliable than on the dataset's cropped grayscale images.
