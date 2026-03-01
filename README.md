# 🤖 GAN-T5 Text Summarizer

A premium text summarization tool powered by **Generative Adversarial Networks** and the **T5 Transformer**.

## 🌟 Features
- **GAN-Enhanced Training:** Uses a discriminator to guide the T5 generator towards more human-like summaries.
- **Abstractive Summarization:** Goes beyond simple extraction to rephrase and condense information.
- **Premium UI:** Beautiful Streamlit-based interface with a dark cosmic theme.
- **Full Analysis:** Detailed ROUGE score comparison between base T5 and the GAN-enhanced version.

## 📁 Project Structure
- `src/`: Core logic (Generator, Discriminator, Training loop).
- `models/`: Directory for saved model weights.
- `docs/`: In-depth analysis and technical documentation.
- `app.py`: Streamlit web application.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face Transformers

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shoaib39011/Text-Summarizer.git
   cd Text-Summarizer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training
To train the model from scratch:
```bash
python src/train.py
```

### Running the App
Launch the interactive web interface:
```bash
streamlit run app.py
```

## 📊 Analysis
Check out the [Analysis Report](docs/analysis.md) for a detailed breakdown of the model's performance and comparison with baseline models.

## 🛠️ Built With
- **T5 (Base Model):** For high-quality text-to-text transformation.
- **DistilBERT (Discriminator):** For efficient adversarial classification.
- **PyTorch:** For the deep learning framework.
- **Streamlit:** For the interactive dashboard.

---
Created by Antigravity AI 🚀
