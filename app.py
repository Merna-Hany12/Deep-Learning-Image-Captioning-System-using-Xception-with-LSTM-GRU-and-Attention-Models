import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────
#  Page Config  (must be FIRST streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PixelNarrate · AI Image Captioning",
    page_icon="🔭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  Custom CSS — Dark editorial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Reset & Base ───────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e4dc;
}

/* ── Hide Streamlit Chrome ───────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 780px; }

/* ── Hero Header ─────────────────────────────── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: #c8a96e;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.6rem, 6vw, 4rem);
    font-weight: 400;
    line-height: 1.08;
    color: #f0ebe0;
    margin: 0 0 0.6rem;
}
.hero-title em {
    font-style: italic;
    color: #c8a96e;
}
.hero-sub {
    font-size: 1rem;
    color: #8a8070;
    font-weight: 300;
    letter-spacing: 0.01em;
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
}
.hero-divider {
    width: 48px;
    height: 1px;
    background: #c8a96e;
    margin: 0 auto 2.5rem;
    opacity: 0.6;
}

/* ── Upload Zone ─────────────────────────────── */
.upload-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #6a6258;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: block;
}

[data-testid="stFileUploader"] {
    border: 1px solid #2a2520 !important;
    border-radius: 4px !important;
    background: #111016 !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #c8a96e !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: 1.5px dashed #2e2a24 !important;
    border-radius: 4px !important;
    color: #6a6258 !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #c8a96e !important;
}

/* ── Displayed Image ─────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 4px;
    border: 1px solid #1e1b16;
    width: 100%;
    max-height: 480px;
    object-fit: contain;
    background: #0d0c10;
}

/* ── Button ──────────────────────────────────── */
.stButton > button {
    width: 100%;
    background: transparent !important;
    border: 1px solid #c8a96e !important;
    color: #c8a96e !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    border-radius: 2px !important;
    transition: all 0.25s ease !important;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    background: #c8a96e !important;
    color: #0a0a0f !important;
}

/* ── Caption Result ──────────────────────────── */
.caption-box {
    margin-top: 2rem;
    padding: 2rem 2rem 1.8rem;
    background: #0e0d12;
    border: 1px solid #1e1b16;
    border-left: 3px solid #c8a96e;
    border-radius: 0 4px 4px 0;
    position: relative;
}
.caption-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    color: #c8a96e;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.caption-text {
    font-family: 'DM Serif Display', serif;
    font-style: italic;
    font-size: 1.45rem;
    line-height: 1.5;
    color: #f0ebe0;
    margin: 0;
}
.caption-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #3a3530;
    margin-top: 1.2rem;
    letter-spacing: 0.1em;
}

/* ── Info Cards ──────────────────────────────── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: #1a1714;
    border: 1px solid #1a1714;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 3rem;
}
.info-card {
    background: #0e0d12;
    padding: 1.4rem 1.2rem;
    text-align: center;
}
.info-card-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #c8a96e;
    margin-bottom: 0.3rem;
}
.info-card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: #4a4540;
    text-transform: uppercase;
}

/* ── Status Messages ─────────────────────────── */
[data-testid="stStatusWidget"],
.stSpinner { color: #8a8070 !important; }

div[data-testid="stNotification"] {
    background: #111016 !important;
    border-color: #2a2520 !important;
    color: #e8e4dc !important;
}

.stSuccess {
    background: #0d1410 !important;
    border-color: #2a4a30 !important;
    color: #6ab870 !important;
}

/* ── Scrollbar ───────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2520; border-radius: 2px; }

/* ── Footer ──────────────────────────────────── */
.app-footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid #1a1714;
}
.app-footer p {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    color: #3a3530;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
IMG_SIZE   = 299
BEAM_WIDTH = 3
MAX_LEN    = None
VOCAB_SIZE = None
word2idx   = {}
idx2word   = {}

VOCAB_PATH   = "vocab.json"
WEIGHTS_PATH = os.path.join("models", "baseline_LSTM.h5")
MODEL_CFG    = {"rnn": "LSTM", "embed": 256, "units": 256, "dropout": 0.0}

# ─────────────────────────────────────────────
#  Vocab loader
# ─────────────────────────────────────────────
def load_vocab(path):
    global MAX_LEN, VOCAB_SIZE, word2idx, idx2word
    with open(path, 'r') as f:
        data = json.load(f)
    word2idx  = data['word2idx']
    idx2word  = {int(k): v for k, v in data['idx2word'].items()}
    VOCAB_SIZE = data['vocab_size']
    MAX_LEN   = data['max_len']

# ─────────────────────────────────────────────
#  Model architecture
# ─────────────────────────────────────────────
def build_tt_lstm(vocab_size, max_len, embed_dim=256, units=256,
                  dropout=0.0, feature_dim=2048):
    img_input = Input(shape=(feature_dim,), name='image_input')
    img_x = layers.Dense(embed_dim, activation='relu', name='img_dense')(img_input)
    img_x = layers.RepeatVector(max_len, name='img_repeat')(img_x)
    img_x = layers.LSTM(units, return_sequences=True, name='img_rnn')(img_x)
    img_x = layers.TimeDistributed(layers.Dense(embed_dim), name='img_td')(img_x)

    seq_input = Input(shape=(max_len,), name='seq_input')
    seq_x = layers.Embedding(vocab_size, embed_dim, mask_zero=False, name='embedding')(seq_input)
    seq_x = layers.LSTM(units, return_sequences=True, name='lang_rnn')(seq_x)
    seq_x = layers.TimeDistributed(layers.Dense(embed_dim), name='lang_td')(seq_x)

    merged = layers.Concatenate(axis=-1, name='concat')([img_x, seq_x])
    dec = layers.Bidirectional(
        layers.LSTM(units, return_sequences=False), name='bidirectional_decoder'
    )(merged)
    dec    = layers.Dense(units, activation='relu', name='dec_dense')(dec)
    output = layers.Dense(vocab_size, activation='softmax', name='output')(dec)

    return Model(inputs=[img_input, seq_input], outputs=output)

@st.cache_resource(show_spinner=False)
def load_captioning_model(weights_path):
    model = build_tt_lstm(VOCAB_SIZE, MAX_LEN,
                          embed_dim=MODEL_CFG['embed'],
                          units=MODEL_CFG['units'])
    model.load_weights(weights_path)
    return model

@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    base  = Xception(weights='imagenet', include_top=False,
                     pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model = tf.keras.Model(inputs=base.input, outputs=base.output)
    model.trainable = False
    return model

# ─────────────────────────────────────────────
#  Inference helpers
# ─────────────────────────────────────────────
def extract_feature(image, extractor):
    img       = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr       = tf.keras.utils.img_to_array(img)
    arr       = np.expand_dims(arr, 0)
    arr       = preprocess_input(arr)
    return extractor.predict(arr, verbose=0)[0]

def beam_search(model, feature, beam_width=BEAM_WIDTH):
    start_id = word2idx.get('startseq', 1)
    end_id   = word2idx.get('endseq',   2)
    pad_id   = word2idx.get('<pad>',    0)
    beams    = [(0.0, [start_id])]
    done     = []

    for _ in range(MAX_LEN):
        candidates = []
        for score, seq in beams:
            if seq[-1] == end_id:
                done.append((score, seq)); continue
            padded  = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
            feat_in = np.expand_dims(feature, 0)
            probs   = model.predict([feat_in, padded], verbose=0)[0]
            top_k   = np.argsort(probs)[-beam_width:]
            for idx in top_k:
                candidates.append((score - np.log(probs[idx] + 1e-10), seq + [idx]))
        if not candidates: break
        candidates.sort(key=lambda x: x[0])
        beams = candidates[:beam_width]

    done += beams
    best = min(done, key=lambda x: x[0])[1]
    words = [idx2word.get(i, '') for i in best
             if i not in (start_id, end_id, pad_id) and idx2word.get(i, '')]
    return ' '.join(words).strip()

# ─────────────────────────────────────────────
#  Hero Section
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI · Computer Vision · NLP</div>
    <h1 class="hero-title">Pixel<em>Narrate</em></h1>
    <p class="hero-sub">Upload any image and let the model describe what it sees — powered by an Xception encoder and a Bidirectional TT-LSTM decoder.</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Load vocab & models (once, silently)
# ─────────────────────────────────────────────
if not os.path.exists(VOCAB_PATH):
    st.error(f"**vocab.json** not found. Place it in the project root.")
    st.stop()

load_vocab(VOCAB_PATH)

with st.spinner("Warming up neural networks…"):
    feature_extractor = load_feature_extractor()

models_ready = os.path.exists(WEIGHTS_PATH)

# ─────────────────────────────────────────────
#  Stat Cards
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="info-grid">
    <div class="info-card">
        <div class="info-card-num">{VOCAB_SIZE:,}</div>
        <div class="info-card-label">Vocab Tokens</div>
    </div>
    <div class="info-card">
        <div class="info-card-num">{MAX_LEN}</div>
        <div class="info-card-label">Max Seq Length</div>
    </div>
    <div class="info-card">
        <div class="info-card-num">B·{BEAM_WIDTH}</div>
        <div class="info-card-label">Beam Width</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Upload
# ─────────────────────────────────────────────
st.markdown('<span class="upload-label">Upload an image</span>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label="Drop or browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    if not models_ready:
        st.error(f"Model weights not found at `{WEIGHTS_PATH}`. "
                 "Place `baseline_LSTM.h5` inside a `models/` directory.")
        st.stop()

    if st.button("✦  Generate Caption"):
        with st.spinner("Extracting visual features…"):
            feat = extract_feature(image, feature_extractor)

        with st.spinner("Decoding with beam search…"):
            caption_model = load_captioning_model(WEIGHTS_PATH)
            caption       = beam_search(caption_model, feat)

        st.markdown(f"""
        <div class="caption-box">
            <div class="caption-label">Generated Caption</div>
            <p class="caption-text">"{caption}"</p>
            <div class="caption-meta">MODEL · baseline_LSTM &nbsp;|&nbsp; BEAM WIDTH · {BEAM_WIDTH} &nbsp;|&nbsp; ENCODER · Xception-299</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <p>PixelNarrate · TT-LSTM Image Captioning · Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)