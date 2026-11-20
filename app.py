import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import os
import docx2txt

# Load the BART model and tokenizer
model_path = "P:\LDS Model"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Define function for text summarization
def generate_summary(text):
    input_ids = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids["input_ids"], num_beams=4, max_length=512, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app
def main():
    # Set background image
    st.markdown(
        """
        <style>
            <link rel="stylesheet" type="text/css" href="P:\LDS Model\background.css">
        }
        </style>
        """
        ,
        unsafe_allow_html=True
    )

    st.title("Welcome to our Legal Document Summariser")
    st.write("Please Upload your document in the file uploader below:")

    # File uploader for document input
    uploaded_file = st.file_uploader("Upload Document", type=['txt', 'pdf', 'docx'])

    # Progress bar for document upload
    progress_bar = st.empty()

    if uploaded_file is not None:
        # Initialize progress bar
        progress_bar.progress(0)
    
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
