import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import os
import docx2txt
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import base64

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the BART model and tokenizer
model_path = "P:\LDS Model"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Define NLTK preprocessing function
def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Optionally, remove stopwords
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        preprocessed_sentences.append(' '.join(filtered_words))
    
    # Optionally, perform stemming
    stemmer = PorterStemmer()
    stemmed_sentences = [stemmer.stem(sentence) for sentence in preprocessed_sentences]
    
    # Join the preprocessed sentences back into a single string
    preprocessed_text = ' '.join(stemmed_sentences)
    
    return preprocessed_text

# Define function for text summarization
def generate_summary(text):
    preprocessed_text = preprocess_text(text)
    input_ids = tokenizer(preprocessed_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids["input_ids"], num_beams=4, max_length=512, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app
def main():
    
    st.title("Welcome to our Legal Document Summariser")
    st.write("Please Upload your document in the file uploader below:")

    # File uploader for document input
    uploaded_file = st.file_uploader("Upload Document", type=['txt', 'pdf', 'docx'])

    # Generate summary button
    if st.button("Generate Summary"):
        if uploaded_file is not None:
            # Read the file
            if uploaded_file.type == 'text/plain':
                text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == 'application/pdf':
                text = " ".join([page.extract_text() for page in uploaded_file.pages])
            elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                text = docx2txt.process(uploaded_file)

            # Generate summary
            summary = generate_summary(text)

            # Display summary
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please upload a file before generating the summary.")

if __name__ == "__main__":
    main()