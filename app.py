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

    You must define in .streamlit/secrets.toml:
      [gcp_service_account]
      ... (full JSON from service account) ...

      GSPREAD_SHEET_ID = "your_google_sheet_id"
    """
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_info, scopes=SHEET_SCOPE)
    client = gspread.authorize(creds)

    sheet_id = st.secrets["GSPREAD_SHEET_ID"]
    sh = client.open_by_key(sheet_id)
    # We'll use the first worksheet in the spreadsheet
    return sh.sheet1


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
    # Ensure sentence_id is string or int consistently
    if "sentence_id" in df.columns:
        df["sentence_id"] = df["sentence_id"].astype(str)
    return df


def append_annotation(sheet, annotator_id: str, sentence_id, label: str):
    """Append a single annotation row to the Google Sheet."""
    timestamp = datetime.utcnow().isoformat()
    row = [annotator_id, str(sentence_id), label, timestamp]
    sheet.append_row(row)


def get_next_sentence_id(data_df: pd.DataFrame, annotations_df: pd.DataFrame, annotator_id: str):
    """
    Return a random sentence_id that this annotator has not yet annotated.
    """
    # All sentence IDs in data
    all_ids = set(data_df["sentence_id"].astype(str).tolist())

    # Sentence_ids already annotated by this annotator
    user_ann = annotations_df[annotations_df["annotator_id"] == annotator_id]
    already_done = set(user_ann["sentence_id"].astype(str).tolist())

    remaining = list(all_ids - already_done)
    if not remaining:
        return None

    return random.choice(remaining)


def choose_new_sentence_id():
    """Helper to update session_state with a new sentence id or None."""
    sheet = get_sheet()
    data_df = load_data()
    ann_df = load_annotations_df(sheet)
    next_id = get_next_sentence_id(data_df, ann_df, st.session_state.annotator_id)
    st.session_state.current_sentence_id = next_id


# ---------------- UI ----------------
st.title("📝 Framing Sentiment Annotation")

st.write(
    """
You will see sentences and be asked to choose whether the **primary sentiment** 
of the sentence is **more positive** or **more negative**.
"""
)

# --- Annotator ID ---
st.markdown("#### Step 1: Identify yourself")

annotator_id = st.text_input(
    "Enter a unique annotator ID (e.g., your email or nickname):",
    help="Use the same ID every time you come back so you won't see the same sentences again.",
)

if not annotator_id:
    st.warning("Please enter your annotator ID to start annotating.")
    st.stop()

# store annotator ID in session state
if "annotator_id" not in st.session_state:
    st.session_state.annotator_id = annotator_id
else:
    st.session_state.annotator_id = annotator_id  # keep updated if they change it

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

# --- Initialize current sentence in session_state ---
if "current_sentence_id" not in st.session_state:
    choose_new_sentence_id()

current_id = st.session_state.current_sentence_id

if current_id is None:
    st.success("🎉 You have annotated all available sentences. Thank you!")
    st.stop()

# Get the row for the current sentence
row = df.loc[df["sentence_id"].astype(str) == str(current_id)]
if row.empty:
    st.error("Could not find the selected sentence in the dataset.")
    st.stop()

sentence_text = row["opposite_framing_sentence"].iloc[0]

st.markdown("---")
st.markdown("### Step 2: Annotate the sentence")

st.markdown(f"**Sentence ID:** `{current_id}`")
st.write(f"**Sentence:** {sentence_text}")

label = st.radio(
    "Overall, what is the **primary sentiment** of this sentence?",
    options=["Positive", "Negative"],
    index=0,
    help="Choose the main sentiment. Don't overthink edge cases – pick the dominant tone.",
)

col1, col2 = st.columns(2)

with col1:
    submitted = st.button("Submit annotation ✅")

with col2:
    skipped = st.button("Skip this sentence ⏭️")

if submitted:
    append_annotation(sheet, annotator_id, current_id, label)
    st.success("Annotation saved. Thank you! 🙌")

    # pick a new sentence and rerun
    choose_new_sentence_id()
    st.rerun()

elif skipped:
    st.info("Sentence skipped. Showing a new one.")
    choose_new_sentence_id()
    st.rerun()

# --- Optional: view/download all annotations ---
st.markdown("---")
st.markdown("### Step 3 (optional): View / download annotations")

if st.checkbox("Show all annotations (may be slow if many rows)"):
    ann_df = load_annotations_df(sheet)
    if ann_df.empty:
        st.info("No annotations yet.")
    else:
        st.write("Showing the last 50 annotations:")
        st.dataframe(ann_df.tail(50))

        csv_bytes = ann_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download all annotations as CSV",
            data=csv_bytes,
            file_name="annotations_export.csv",
            mime="text/csv",
        )
