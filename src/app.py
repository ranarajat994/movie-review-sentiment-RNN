# ============================================================
# Movie Review Sentiment Analysis - app.py
# Run from project root:  streamlit run src/app.py
# ============================================================

# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------
import streamlit as st

from predict import predict_sentiment
from predict import tokenizer
from predict import settings


# ------------------------------------------------------------
# 2. Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="ReelFeel · Movie Review Sentiment",
    page_icon="🎬",
    layout="centered",
)

MAX_WORDS = settings["max_words"]
MAX_LEN = settings["max_len"]


# ------------------------------------------------------------
# 3. Custom styling (CSS) - cinema theme
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, label, textarea, button {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: radial-gradient(circle at 50% -10%, #2a2233 0%, #141118 55%, #0e0c11 100%);
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 820px; }

/* ---------- Hero (marquee) ---------- */
.marquee {
    text-align: center;
    border: 2px solid #f5b942;
    border-radius: 18px;
    padding: 26px 20px 22px 20px;
    margin-bottom: 26px;
    position: relative;
    background: rgba(245, 185, 66, 0.04);
    box-shadow: 0 0 40px rgba(245, 185, 66, 0.08);
}
.marquee::before, .marquee::after {
    content: "● ● ● ● ● ● ● ● ● ● ● ● ● ●";
    position: absolute; left: 0; right: 0;
    color: #f5b942; opacity: 0.55; font-size: 9px; letter-spacing: 9px;
}
.marquee::before { top: 7px; }
.marquee::after { bottom: 7px; }
.marquee .tag {
    color: #f5b942; font-size: 12px; letter-spacing: 3px; font-weight: 700; text-transform: uppercase;
}
.marquee h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 64px; letter-spacing: 3px; color: #fbf3e4; margin: 4px 0 2px 0; line-height: 1;
}
.marquee p { color: #b9aec4; font-size: 15px; margin: 6px 0 0 0; }

/* ---------- Stats strip ---------- */
.stats { display: flex; justify-content: center; gap: 12px; margin: -6px 0 22px 0; flex-wrap: wrap; }
.stat {
    background: #1e1a24; border: 1px solid #2f2838; border-radius: 12px;
    padding: 10px 16px; text-align: center; min-width: 120px;
}
.stat b { display: block; color: #fbf3e4; font-size: 20px; }
.stat span { color: #8e839b; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }

/* ---------- Labels and text area ---------- */
.label { color: #fbf3e4; font-weight: 600; font-size: 15px; margin-bottom: 6px; }
.hint { color: #8e839b; font-size: 13px; margin-bottom: 8px; }
.stTextArea textarea {
    background: #1b1720 !important; color: #fbf3e4 !important;
    border: 1px solid #3a3145 !important; border-radius: 14px !important;
    font-size: 15px !important; line-height: 1.55 !important;
}
.stTextArea textarea:focus { border-color: #f5b942 !important; box-shadow: 0 0 0 1px #f5b942 !important; }

/* ---------- Buttons ---------- */
.stButton button {
    background: #1e1a24; color: #e5dcef; border: 1px solid #3a3145;
    border-radius: 999px; font-size: 13px; padding: 4px 14px;
}
.stButton button:hover { border-color: #f5b942; color: #f5b942; }
div[data-testid="stButton"] button[kind="primary"] {
    background: #f5b942; color: #141118; border: none; font-weight: 700;
    font-size: 16px; padding: 12px 0; border-radius: 12px;
}
div[data-testid="stButton"] button[kind="primary"]:hover { background: #ffd27a; color: #141118; }

/* ---------- Result card ---------- */
.result {
    border-radius: 20px; padding: 26px 26px 22px 26px; margin-top: 22px;
    text-align: center; border: 1px solid;
}
.result-pos { background: linear-gradient(160deg, rgba(56,176,110,0.16), rgba(56,176,110,0.04)); border-color: rgba(56,176,110,0.45); }
.result-neg { background: linear-gradient(160deg, rgba(226,75,74,0.16), rgba(226,75,74,0.04)); border-color: rgba(226,75,74,0.45); }
.result .emoji { font-size: 52px; line-height: 1; }
.result .verdict {
    font-family: 'Bebas Neue', sans-serif; font-size: 50px; letter-spacing: 3px; margin: 6px 0 0 0;
}
.result-pos .verdict { color: #5fd394; }
.result-neg .verdict { color: #ff7b7a; }
.result .strength { color: #cfc5d9; font-size: 15px; margin-top: 2px; }

/* ---------- Sentiment meter ---------- */
.meter-wrap { margin: 22px 4px 6px 4px; }
.meter {
    position: relative; height: 12px; border-radius: 999px;
    background: linear-gradient(90deg, #e24b4a 0%, #8a7f96 50%, #38b06e 100%);
}
.meter-dot {
    position: absolute; top: 50%; width: 22px; height: 22px; border-radius: 50%;
    background: #fbf3e4; border: 3px solid #141118; transform: translate(-50%, -50%);
    box-shadow: 0 0 0 2px #fbf3e4;
}
.meter-labels { display: flex; justify-content: space-between; color: #8e839b; font-size: 12px; margin-top: 8px; }

/* ---------- What the model saw ---------- */
.section { color: #fbf3e4; font-weight: 700; font-size: 16px; margin: 26px 0 4px 0; }
.chips { line-height: 2.2; margin-top: 6px; }
.chip {
    display: inline-block; padding: 1px 10px; border-radius: 8px; margin: 0 4px 0 0;
    font-size: 13px; background: #2a2433; color: #e5dcef;
}
.chip-neg { background: rgba(245,185,66,0.18); color: #f5b942; font-weight: 600; }
.chip-oov { background: transparent; color: #6f6579; border: 1px dashed #4a4055; }
.mini-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }
.mini { background: #1e1a24; border: 1px solid #2f2838; border-radius: 12px; padding: 12px; }
.mini b { display: block; color: #fbf3e4; font-size: 20px; }
.mini span { color: #8e839b; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.legend { color: #8e839b; font-size: 12px; margin-top: 10px; }

/* ---------- Footer ---------- */
.footer { text-align: center; color: #6f6579; font-size: 13px; margin-top: 40px; }
.footer a { color: #f5b942; text-decoration: none; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 4. Hero banner
# ------------------------------------------------------------
st.markdown("""
<div class="marquee">
    <div class="tag">Now showing · Deep Learning</div>
    <h1>REELFEEL</h1>
    <p>Paste any movie review and an LSTM network will tell you if it's a rave or a rant.</p>
</div>
""", unsafe_allow_html=True)

test_acc = round(settings["test_accuracy"] * 100, 1)

st.markdown(f"""
<div class="stats">
    <div class="stat"><b>{test_acc}%</b><span>Test accuracy</span></div>
    <div class="stat"><b>{settings["best_model"]}</b><span>Best of 3 RNNs</span></div>
    <div class="stat"><b>50K</b><span>IMDB reviews</span></div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 5. Example reviews (buttons fill the text box)
# ------------------------------------------------------------
example_positive = ("An absolute masterpiece. The performances are stunning, the music gave me chills, "
                    "and the ending left the whole theatre in tears. I would watch it again tomorrow.")
example_negative = ("What a waste of two hours. The plot made no sense, the dialogue was painful "
                    "and I nearly fell asleep halfway through. Avoid this one.")
example_tricky = ("I expected to hate this movie, but honestly it wasn't bad at all. "
                  "Not perfect, but the cast clearly had fun and so did I.")

if "review" not in st.session_state:
    st.session_state["review"] = ""


def use_positive():
    st.session_state["review"] = example_positive


def use_negative():
    st.session_state["review"] = example_negative


def use_tricky():
    st.session_state["review"] = example_tricky


def clear_review():
    st.session_state["review"] = ""


# ------------------------------------------------------------
# 6. Input
# ------------------------------------------------------------
st.markdown('<div class="label">Your review</div>', unsafe_allow_html=True)
st.markdown('<div class="hint">Type your own, or try one of the examples.</div>', unsafe_allow_html=True)

b1, b2, b3, b4 = st.columns([1.1, 1.1, 1.3, 0.8])
with b1:
    st.button("😍 A rave", on_click=use_positive, use_container_width=True)
with b2:
    st.button("😡 A rant", on_click=use_negative, use_container_width=True)
with b3:
    st.button("🤔 A tricky one", on_click=use_tricky, use_container_width=True)
with b4:
    st.button("Clear", on_click=clear_review, use_container_width=True)

review = st.text_area("Your review", key="review", height=170,
                      placeholder="e.g. The story was slow at first, but the last hour was brilliant...",
                      label_visibility="collapsed")

analyze = st.button("🎬  Get the verdict", type="primary", use_container_width=True)


# ------------------------------------------------------------
# 7. Prediction and result
# ------------------------------------------------------------
if analyze:

    if len(review.strip().split()) < 3:
        st.warning("Please write at least a few words so the model has something to read.")

    else:
        sentiment, confidence, probability, cleaned = predict_sentiment(review)
        confidence_pct = round(confidence * 100, 1)

        # How sure is the model?
        if confidence >= 0.85:
            strength = "Strong"
        elif confidence >= 0.65:
            strength = "Moderate"
        else:
            strength = "Uncertain"

        if sentiment == "Positive":
            card_class = "result-pos"
            emoji = "🍿"
            verdict = "Positive review"
        else:
            card_class = "result-neg"
            emoji = "🍅"
            verdict = "Negative review"

        dot_position = round(probability * 100, 1)

        st.markdown(f"""
        <div class="result {card_class}">
            <div class="emoji">{emoji}</div>
            <div class="verdict">{verdict}</div>
            <div class="strength">{strength} confidence · <b>{confidence_pct}%</b></div>
            <div class="meter-wrap">
                <div class="meter"><div class="meter-dot" style="left: {dot_position}%;"></div></div>
                <div class="meter-labels"><span>Negative</span><span>Neutral</span><span>Positive</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if strength == "Uncertain":
            st.info("The model isn't sure about this one. Mixed reviews, sarcasm and heavy negation "
                    "are the hardest cases for an LSTM trained from scratch.")

        # --------------------------------------------------------
        # What the model actually saw
        # --------------------------------------------------------
        cleaned_words = cleaned.split()
        original_count = len(review.split())
        kept_count = len(cleaned_words)

        negation_set = {"not", "no", "nor", "but", "never"}

        # build one coloured chip per word
        chips_html = ""
        unknown_count = 0
        for word in cleaned_words:
            word_index = tokenizer.word_index.get(word, MAX_WORDS + 1)
            if word_index >= MAX_WORDS:
                chips_html += f'<span class="chip chip-oov">{word}</span>'
                unknown_count += 1
            elif word in negation_set:
                chips_html += f'<span class="chip chip-neg">{word}</span>'
            else:
                chips_html += f'<span class="chip">{word}</span>'

        if kept_count > MAX_LEN:
            truncated_note = f"<br>Long review: only the last {MAX_LEN} words were used (the ending usually holds the verdict)."
        else:
            truncated_note = ""

        st.markdown('<div class="section">What the model actually read</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="chips">{chips_html}</div>
        <div class="legend">
            <span class="chip chip-neg">not</span> negation words were kept on purpose ·
            <span class="chip chip-oov">word</span> outside the model's 10,000-word vocabulary
            {truncated_note}
        </div>
        <div class="mini-grid">
            <div class="mini"><span>Words you wrote</span><b>{original_count}</b></div>
            <div class="mini"><span>After cleaning</span><b>{kept_count}</b></div>
            <div class="mini"><span>Unknown to model</span><b>{unknown_count}</b></div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------
# 8. How it works
# ------------------------------------------------------------
st.write("")
with st.expander("🎞️  How it works"):
    st.markdown(f"""
**1. Cleaning** — lowercase, remove HTML tags and URLs with regex, expand *didn't → did not*,
remove stopwords (but **keep negations** like *not*, *no*, *but*), and lemmatize (*movies → movie*).

**2. Tokenizing** — Keras `Tokenizer` keeps the top **{MAX_WORDS:,}** words; anything else becomes `<OOV>`.

**3. Padding** — every review becomes exactly **{MAX_LEN}** numbers (`padding='pre'`, `truncating='pre'`),
so the ending of long reviews is kept.

**4. Model** — Embedding (64) → **{settings["best_model"]}** (64 units) → Dropout (0.3) → Dense → Sigmoid,
with early stopping.

| Model | Test accuracy |
|---|---|
| SimpleRNN | 84.9% |
| **LSTM** | **88.9%** |
| GRU | 88.5% |

SimpleRNN struggles with long reviews (vanishing gradients); LSTM and GRU use gates to remember
what matters across 200 words.
""")


# ------------------------------------------------------------
# 9. Footer
# ------------------------------------------------------------
st.markdown("""
<div class="footer">
    Built by <b>Rajat Rana</b> ·
    <a href="https://github.com/ranarajat994" target="_blank">GitHub</a> ·
    <a href="https://linkedin.com/in/rajat-rana-86365991" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
