import streamlit as st
import re
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline

# Download punkt if needed
nltk.download('punkt')

# Text Preprocessing
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    sentences = sent_tokenize(text)
    return " ".join(sentences)

# Extractive Summarization
def extractive_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Abstractive Summarization
abstractive_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def abstractive_summary(text):
    summary = abstractive_summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Streamlit UI
st.title("🧠 Text Summarization App")
st.markdown("Summarize long documents using **Extractive** or **Abstractive** methods.")

text_input = st.text_area("✏️ Enter text to summarize:", height=300)
method = st.radio("Choose summarization type:", ["Extractive", "Abstractive"])

if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            processed = preprocess_text(text_input)
            if method == "Extractive":
                result = extractive_summary(processed)
            else:
                result = abstractive_summary(processed)
        st.subheader("📝 Summary")
        st.write(result)
