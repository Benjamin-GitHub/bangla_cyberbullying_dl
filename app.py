# app.py

from pathlib import Path

import streamlit as st
import pandas as pd

from src.inference import predict_label

st.set_page_config(
    page_title="Bangla Cyberbullying Detector (BiLSTM)",
    page_icon="🛡️",
)

DEMO_DATA_PATH = Path("data/demo_examples.csv")


@st.cache_data
def load_demo_dataset() -> pd.DataFrame | None:
    """
    Load a small demo dataset (around 200 examples) from data/demo_examples.csv.

    The CSV is expected to contain at least a 'text' column, and optionally a 'label' column.
    """
    if not DEMO_DATA_PATH.exists():
        return None

    df = pd.read_csv(DEMO_DATA_PATH)
    
    if "text" not in df.columns:
        return None

    return df


# One example per class
EXAMPLE_COMMENTS = {
    "Neutral": "আজকে আবহাওয়া খুব সুন্দর, সবাইকে অনেক শুভেচ্ছা।",
    "Political": "তোমার রাজনৈতিক দল সব সময় মিথ্যা কথা বলে, কারও ভাল চায় না।",
    "Sexual": "গরীবের ভন্ডু জনি সিন্স তোমার পাশে দাড়াবে চিন্তা কইর না",
    "Threat": "আর একবার এমন করলে কিন্তু ভালো হবে না, মনে রাখবে।",
    "Troll": "তুমি নাকি বড় জ্ঞানী! হাহা, কিছুই তো জানো না।",
}


if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

st.title("🛡️ Bangla Cyberbullying Detector (BiLSTM)")
st.write(
    """
Paste a Bangla social media comment below.  
The model will predict whether it is cyberbullying and which **category** it belongs to
(political, sexual, troll, threat, or neutral).
"""
)

st.markdown("---")

# Sidebar: demo dataset with ~200 examples
demo_df = load_demo_dataset()

if demo_df is not None:
    st.sidebar.subheader("Demo dataset (examples)")
    st.sidebar.write(f"{len(demo_df)} sample comments loaded.")

    # Let the observer pick a row index
    max_index = len(demo_df) - 1
    selected_index = st.sidebar.number_input(
        "Select example row index",
        min_value=0,
        max_value=max_index,
        value=0,
        step=1,
    )

    if st.sidebar.button("Load example into text box"):
        # Expect a 'text' column in the CSV
        st.session_state["input_text"] = str(demo_df.iloc[selected_index]["text"])

    if "label" in demo_df.columns:
        st.sidebar.caption(
            "Dataset includes a 'label' column (ground truth category) for each comment."
        )

    with st.sidebar.expander("Show demo dataset preview"):
        st.dataframe(demo_df.head(200))
else:
    st.sidebar.subheader("Demo dataset")
    st.sidebar.info(
        "To enable demo examples, add a CSV at 'data/demo_examples.csv' "
        "with at least a 'text' column (and optional 'label' column)."
    )

# Built-in example selector (one per class)
example_choice = st.selectbox(
    "Or choose an example comment (one per class)",
    ["(none)"] + list(EXAMPLE_COMMENTS.keys()),
)

if example_choice != "(none)":
    st.session_state["input_text"] = EXAMPLE_COMMENTS[example_choice]

# Text input
user_text = st.text_area(
    "Bangla comment",
    placeholder="(Paste a Bangla comment here)...",
    height=160,
    key="input_text",
)

col_left, col_right = st.columns([1, 3])

with col_left:
    classify_button = st.button("Classify")

if classify_button:
    if not user_text.strip():
        st.warning("Please enter a comment first.")
    else:
        try:
            label, best_prob, probs = predict_label(user_text)
        except ValueError as e:
            st.error(str(e))
        else:
            # Main prediction
            st.subheader("Prediction")
            st.write(
                f"**Predicted label:** `{label}`  "
                f"({best_prob * 100:.1f}% confidence)"
            )

            # Probability table
            st.subheader("Class probabilities")

            # Sort by highest probability first
            sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)

            df = pd.DataFrame(
                {
                    "Label": [k for k, _ in sorted_items],
                    "Probability": [float(v) for _, v in sorted_items],
                    "Percentage": [round(float(v) * 100, 2) for _, v in sorted_items],
                }
            )

            st.table(df)

            st.caption(
                "⚠️ This tool is for research/educational use only. "
                "Decisions about content moderation should always involve human review."
            )