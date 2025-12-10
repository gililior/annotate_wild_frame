import streamlit as st
import pandas as pd
import random
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

# ---------------- CONFIG ----------------
DATA_PATH = "framed_eval.csv"  # your dataset
SHEET_SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]

st.set_page_config(
    page_title="Sentiment Annotation",
    page_icon="📝",
    layout="centered",
)

# Global CSS for larger fonts
st.markdown(
    """
    <style>
    .sentence-text {
        font-size: 1.4rem;
        font-weight: 500;
        line-height: 1.6;
    }
    div.stRadio > div[role='radiogroup'] label {
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HELPERS ----------------
@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = ["sentence_id", "opposite_framing_sentence"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {path}")
    return df


@st.cache_resource
def get_sheet():
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_info, scopes=SHEET_SCOPE)
    client = gspread.authorize(creds)

    sheet_id = st.secrets["GSPREAD_SHEET_ID"]
    sh = client.open_by_key(sheet_id)
    return sh.sheet1


def load_annotations_df(sheet) -> pd.DataFrame:
    records = sheet.get_all_records()
    if not records:
        return pd.DataFrame(columns=["annotator_id", "sentence_id", "label", "timestamp", "label_order_first"])
    df = pd.DataFrame(records)
    if "sentence_id" in df.columns:
        df["sentence_id"] = df["sentence_id"].astype(str)
    return df


def append_annotation(sheet, annotator_id, sentence_id, label, label_order):
    timestamp = datetime.utcnow().isoformat()
    row = [annotator_id, str(sentence_id), label, timestamp, label_order]
    sheet.append_row(row)


def get_next_sentence_id(data_df, annotations_df, annotator_id):
    all_ids = set(data_df["sentence_id"].astype(str).tolist())
    user_ann = annotations_df[annotations_df["annotator_id"] == annotator_id]
    already_done = set(user_ann["sentence_id"].astype(str).tolist())

    remaining = list(all_ids - already_done)
    if not remaining:
        return None
    return random.choice(remaining)


def choose_new_sentence_id():
    sheet = get_sheet()
    data_df = load_data()
    ann_df = load_annotations_df(sheet)
    next_id = get_next_sentence_id(data_df, ann_df, st.session_state.annotator_id)
    st.session_state.current_sentence_id = next_id


def get_user_progress(data_df, annotations_df, annotator_id):
    total = len(data_df)
    user_ann = annotations_df[annotations_df["annotator_id"] == annotator_id]
    done = len(user_ann)
    return done, total


def assign_random_label_order():
    """Randomize whether Positive or Negative appears first for this user."""
    return random.sample(["Positive", "Negative"], k=2)


# ---------------- UI FLOW ----------------

# --- Step 0: Name / ID page ---
if "annotator_id" not in st.session_state:
    st.title("📝 Sentiment Annotation")

    st.markdown("### Step 1: Identify yourself")

    name = st.text_input(
        "Enter a unique annotator ID (your name, email, or nickname):",
        help="Use the same ID every time so you won't see repeated sentences."
    )

    start = st.button("Start annotating ➡️")

    if start:
        if not name.strip():
            st.warning("Please enter a valid annotator ID before starting.")
            st.stop()
        st.session_state.annotator_id = name.strip()

        # assign random order once per user
        st.session_state.label_order = assign_random_label_order()
        st.rerun()
    else:
        st.stop()


# If we’re here, annotator is identified
annotator_id = st.session_state.annotator_id
if "label_order" not in st.session_state:
    st.session_state.label_order = assign_random_label_order()

st.title("📝 Sentiment Annotation")

# --- Load data + sheet ---
df = load_data()
sheet = get_sheet()
ann_df = load_annotations_df(sheet)

# --- Progress bar ---
done, total = get_user_progress(df, ann_df, annotator_id)
total = min(total, 100)
progress = done / total if total > 0 else 0

st.markdown(f"**Annotator ID:** `{annotator_id}`")
st.progress(progress)
st.caption(f"You have annotated {done} sentences.")

# --- Select next sentence ---
if "current_sentence_id" not in st.session_state:
    choose_new_sentence_id()

current_id = st.session_state.current_sentence_id

if current_id is None:
    st.success("🎉 You have annotated all available sentences. Thank you!")
    st.stop()

row = df.loc[df["sentence_id"].astype(str) == str(current_id)]
sentence_text = row["opposite_framing_sentence"].iloc[0]

# --- Instructions ---
st.markdown("### Instructions")
st.write(
    """
You will be provided with a sentence.  
Your job is to annotate what is the **primary sentiment** that is reflected from the sentence.
"""
)

# --- Sentence ---
st.markdown("### Sentence")
st.markdown(
    f"<div class='sentence-text'>{sentence_text}</div>",
    unsafe_allow_html=True,
)

# --- Annotation ---
label_order = st.session_state.label_order  # e.g. ["Positive", "Negative"]
label_order_first = label_order[0]

label = st.radio(
    "",
    options=label_order,
    index=None,  # no default
    key=f"sentiment_choice_{current_id}",  # fresh widget per sentence
)

submitted = st.button("Submit annotation ✅")

if submitted:
    if label is None:
        st.warning("Please choose a sentiment before submitting.")
        st.stop()

    append_annotation(
        sheet,
        annotator_id,
        current_id,
        label,
        label_order_first  # NEW: log which option appeared first
    )

    st.success("Annotation saved. Thank you! 🙌")

    choose_new_sentence_id()
    st.rerun()
