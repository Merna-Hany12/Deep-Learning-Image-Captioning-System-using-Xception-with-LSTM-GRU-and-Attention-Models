# PixelNarrate — AI Image Captioning

> **Describe the world, one image at a time.**  
> An end-to-end image captioning system powered by an Xception visual encoder and a Bidirectional Twin-Tower LSTM (TT-LSTM) language decoder.

---

## 🔭 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bhzqi9j9qk6pwm5zjmz3b8.streamlit.app/)

> 🚧 **To deploy your own copy**, follow the [Deployment](#-deployment) section below.  
> Once deployed, replace the badge URL above with your own Streamlit Cloud link.

**Demo Walkthrough:**

```
1. Open the app
2. Upload any .jpg / .png / .webp image
3. Click "✦ Generate Caption"
4. Read your AI-generated description in under ~5 seconds
```

## ✨ Features

- **One-click captioning** — upload an image, get a natural-language description instantly
- **Xception encoder** — ImageNet-pretrained CNN extracts rich 2048-dim visual features at 299×299 resolution
- **TT-LSTM decoder** — Twin-Tower architecture processes image and language streams independently, then merges via a Bidirectional LSTM decoder
- **Beam search** — width-3 beam search for higher-quality, more coherent captions over greedy decoding
- **Clean dark UI** — editorial aesthetic built with custom CSS; mobile-friendly layout
- **Fully cached** — feature extractor and model are loaded once and cached across sessions via `@st.cache_resource`

---

## 🏗️ Architecture

```
Image (RGB)
    │
    ▼
┌─────────────────────────────────────────────────┐
│              Xception Encoder                   │
│  Input: 299×299×3  →  Output: 2048-dim vector  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────┐   ┌──────────────────────┐
│   Image Tower        │   │   Language Tower     │
│  Dense(256) → ReLU   │   │  Embedding(256)      │
│  RepeatVector(MAX_L) │   │  LSTM(256, seq=True) │
│  LSTM(256, seq=True) │   │  TimeDistributed     │
│  TimeDistributed     │   │  Dense(256)          │
│  Dense(256)          │   └──────────────────────┘
└──────────────────────┘            │
            │                       │
            └──────────┬────────────┘
                       │  Concatenate
                       ▼
            ┌─────────────────────────┐
            │  Bidirectional LSTM(256)│
            │  Dense(256) → ReLU      │
            │  Dense(VOCAB) → Softmax │
            └─────────────────────────┘
                       │
                       ▼
              Caption token (t)
```

**Decoding strategy:** beam search with width = 3, sequence length capped at `MAX_LEN`.

---

## 📁 Project Structure

```
pixelnarrate/
├── app.py                  # Streamlit application
├── vocab.json              # Vocabulary — word2idx, idx2word, vocab_size, max_len
├── models/
│   └── baseline_LSTM.h5    # Trained model weights
├── requirements.txt        # Python dependencies
├── Image_Captioning_Xception_Experiments.ipynb 
└── README.md
```

---

## 📓 Companion Notebook

[`Image_Captioning_Xception_Experiments.ipynb`](./Image_Captioning_Xception_Experiments.ipynb) presents the complete pipeline for building and evaluating an image captioning system using deep learning. It compares three architectures (LSTM baseline, GRU, and attention‑based models) on top of Xception CNN features.

### 🔬 Key Steps

- Image preprocessing (Xception‑compatible 299×299 resizing)
- Text tokenization and sequence padding
- Feature extraction using a frozen pretrained Xception (2048‑dim vectors)
- Training of three models: LSTM (baseline), GRU, and Bahdanau attention
- Evaluation using **BLEU-1/2/3/4** and **ROUGE-L**

### 📊 Selected Results (BLEU‑4, Flickr8k test set)

| Model | BLEU‑4 |
|-------|--------|
| Baseline LSTM | 16.10 |
| GRU + Attention | 15.82 |
| Fine‑tuned Xception + Attention* | 12.62 |

> *Limited to 5 epochs → under‑trained. The notebook discusses how longer training stabilises attention and improves performance.

### ⚠️ Error Analysis

The notebook documents common failure modes:

- Missing objects (e.g., “ball” omitted from a football scene)
- Generic or repetitive captions (“a person is standing” repeated)
- Incorrect object relationships or attributes (wrong colour, size, or action)

### 🎯 Outcome

Among the three architectures, **attention‑based models produce the most accurate and descriptive captions** when adequately trained. The notebook provides a full comparison of trade‑offs between BLEU (exact n‑gram overlap) and ROUGE (semantic similarity).


---


### Required files

| File | Description |
|---|---|
| `vocab.json` | JSON with `word2idx`, `idx2word`, `vocab_size`, `max_len` keys |
| `models/baseline_LSTM.h5` | Keras model weights (saved via `model.save_weights(...)`) |

---

## ⚙️ Local Setup

### Prerequisites

- Python 3.9 – 3.11
- pip

### 1 — Clone the repository

```bash
git clone https://github.com/Merna-Hany12/Deep-Learning-Image-Captioning-System-using-Xception-with-LSTM-GRU-and-Attention-Models.git
cd pixelnarrate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Add model files

Place your trained files in the project root:

```
pixelnarrate/
├── vocab.json
└── models/
    └── baseline_LSTM.h5
```

### 4 — Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## ☁️ Deployment

### Streamlit Community Cloud (recommended — free)

1. Push this repository to GitHub (include `vocab.json` and `models/baseline_LSTM.h5`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch (`main`), and set **Main file path** to `app.py`
4. Click **Deploy** — your app will be live in ~2 minutes

> **Model file size:** Streamlit Cloud supports repos up to 1 GB. If your `.h5` file is large, consider storing it in [Git LFS](https://git-lfs.com/) or loading it from an external URL at startup.

### Alternative: Hugging Face Spaces

1. Create a new Space → select **Streamlit** SDK
2. Upload all project files via the web UI or `git push`
3. Your app is live at `https://huggingface.co/spaces/your-username/pixelnarrate`

---

## 🗂️ vocab.json Format

The vocabulary file must follow this exact schema:

```json
{
  "word2idx": {
    "startseq": 1,
    "endseq": 2,
    "a": 3,
    "dog": 4,
    ...
  },
  "idx2word": {
    "1": "startseq",
    "2": "endseq",
    "3": "a",
    "4": "dog",
    ...
  },
  "vocab_size": 8256,
  "max_len": 34
}
```

---

## 🔬 Model Details

| Parameter | Value |
|---|---|
| Architecture | TT-LSTM (Twin-Tower LSTM) |
| Visual encoder | Xception (ImageNet, frozen) |
| Input resolution | 299 × 299 px |
| Feature dimension | 2048 |
| Embedding dim | 256 |
| LSTM units | 256 |
| Decoder | Bidirectional LSTM |
| Beam width | 3 |
| Training dataset | MS-COCO / Flickr30k *(specify yours)* |

---

## 🧩 How It Works

1. **Feature extraction** — the uploaded image is resized to 299×299, preprocessed with Xception's `preprocess_input`, and passed through the frozen Xception backbone to produce a single 2048-dim vector.

2. **Caption generation** — at each timestep, the model receives the image feature and the partial caption sequence. The image tower repeats the feature across all time steps and processes it with an LSTM; the language tower encodes the token sequence with an Embedding layer and LSTM. The two streams are concatenated and decoded by a Bidirectional LSTM, producing a probability distribution over the vocabulary.

3. **Beam search** — instead of greedily picking the top token at each step, beam search maintains the top-3 candidate sequences and returns the one with the highest cumulative log-probability at the end.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

```bash
# Run a quick sanity check after any code change
streamlit run app.py
```

---

## 📄 License

MIT © 2024 — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ♥ using TensorFlow · Keras · Streamlit</sub>
</div>
