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
    page_title="Framing Sentiment Annotation",
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
    /* Make radio labels bigger */
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
    """Load the dataset with sentences to annotate."""
    df = pd.read_csv(path)
    required_cols = ["sentence_id", "opposite_framing_sentence"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {path}")
    return df


@st.cache_resource
def get_sheet():
    """
    Connect to Google Sheets using service account info from Streamlit secrets.

    In .streamlit/secrets.toml (or Streamlit Cloud secrets) you must define:

      GSPREAD_SHEET_ID = "your_google_sheet_id"

      [gcp_service_account]
      ... full JSON from the service account key ...
    """
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_info, scopes=SHEET_SCOPE)
    client = gspread.authorize(creds)

    sheet_id = st.secrets["GSPREAD_SHEET_ID"]
    sh = client.open_by_key(sheet_id)
    return sh.sheet1  # first worksheet


def load_annotations_df(sheet) -> pd.DataFrame:
    """
    Load all annotations from the Google Sheet into a DataFrame.
    Assumes the first row of the sheet contains headers:
      annotator_id | sentence_id | label | timestamp
    """
    records = sheet.get_all_records()  # list of dicts
    if not records:
        return pd.DataFrame(columns=["annotator_id", "sentence_id", "label", "timestamp"])
    df = pd.DataFrame(records)
    if "sentence_id" in df.columns:
        df["sentence_id"] = df["sentence_id"].astype(str)
    return df


def append_annotation(sheet, annotator_id: str, sentence_id, label: str):
    """Append a single annotation row to the Google Sheet."""
    timestamp = datetime.utcnow().isoformat()
    row = [annotator_id, str(sentence_id), label, timestamp]
    sheet.append_row(row)


def get_next_sentence_id(
    data_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    annotator_id: str
):
    """Return a random sentence_id that this annotator has not yet annotated."""
    all_ids = set(data_df["sentence_id"].astype(str).tolist())

    user_ann = annotations_df[annotations_df["annotator_id"] == annotator_id]
    already_done = set(user_ann["sentence_id"].astype(str).tolist())

    remaining = list(all_ids - already_done)
    if not remaining:
        return None

    return random.choice(remaining)


def choose_new_sentence_id():
    """Update session_state.current_sentence_id with a new unseen id or None."""
    sheet = get_sheet()
    data_df = load_data()
    ann_df = load_annotations_df(sheet)
    next_id = get_next_sentence_id(data_df, ann_df, st.session_state.annotator_id)
    st.session_state.current_sentence_id = next_id


def get_user_progress(data_df: pd.DataFrame, annotations_df: pd.DataFrame, annotator_id: str):
    """Return (num_annotated_by_user, total_sentences)."""
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
    st.title("📝 Framing Sentiment Annotation")

    st.markdown("### Step 1: Identify yourself")

    name = st.text_input(
        "Enter a unique annotator ID (e.g., your name, email or nickname):",
        help="Use the same ID every time you come back so you won't see the same sentences again."
    )

    start = st.button("Start annotating ➡️")

    if start:
        if not name.strip():
            st.warning("Please enter a valid annotator ID before starting.")
            st.stop()
        st.session_state.annotator_id = name.strip()
        # decide random label order for this user once here
        st.session_state.label_order = assign_random_label_order()
        st.rerun()
    else:
        st.stop()

# If we are here, annotator_id already set, and we are on the "main page"
annotator_id = st.session_state.annotator_id

# ensure label_order exists (for safety if reload)
if "label_order" not in st.session_state:
    st.session_state.label_order = assign_random_label_order()

st.title("📝 Framing Sentiment Annotation")

# --- Load data + sheet ---
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

try:
    sheet = get_sheet()
except Exception as e:
    st.error(f"Error connecting to Google Sheets: {e}")
    st.stop()

# Load annotations for progress and selection
ann_df = load_annotations_df(sheet)

# --- Progress bar ---
done, total = get_user_progress(df, ann_df, annotator_id)
progress = done / total if total > 0 else 0.0

st.markdown(f"**Annotator ID:** `{annotator_id}`")
st.progress(progress)
st.caption(f"You have annotated {done} out of {total} sentences.")

# --- Initialize current sentence in session_state ---
if "current_sentence_id" not in st.session_state:
    choose_new_sentence_id()

current_id = st.session_state.current_sentence_id

if current_id is None:
    st.success("🎉 You have annotated all available sentences. Thank you!")
    st.stop()

# Get current sentence
row = df.loc[df["sentence_id"].astype(str) == str(current_id)]
if row.empty:
    st.error("Could not find the selected sentence in the dataset.")
    st.stop()

sentence_text = row["opposite_framing_sentence"].iloc[0]

# --- Instructions + sentence ---
st.markdown("### Instructions")
st.write(
    """
You will be provided with a sentence.  
Your job is to annotate what is the **primary sentiment** that is reflected from the sentence.
"""
)

st.markdown("### Sentence")
st.markdown(
    f"<div class='sentence-text'>{sentence_text}</div>",
    unsafe_allow_html=True,
)

# --- Annotation (no extra title, bigger fonts via CSS) ---
label = st.radio(
    "Overall, what is the **primary sentiment** of this sentence?",
    options=st.session_state.label_order,
    index=None,  # no default selection
    key=f"sentiment_choice_{current_id}",  # fresh widget per sentence
    help="Choose the main sentiment expressed in the sentence."
)

submitted = st.button("Submit annotation ✅")

if submitted:
    if label is None:
        st.warning("Please choose a sentiment before submitting.")
        st.stop()

    # Save annotation with the logical label name ("Positive" / "Negative")
    append_annotation(sheet, annotator_id, current_id, label)

    st.success("Annotation saved. Thank you! 🙌")

    # Pick a new sentence and rerun
    choose_new_sentence_id()
    st.rerun()
