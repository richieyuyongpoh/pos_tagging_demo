import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import contextlib

# Download the necessary NLTK models and corpora
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to perform POS tagging
def pos_tagging(text):
    nltk_tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(nltk_tokens)
    return tagged

# Function to generate a word cloud from POS tags
def generate_wordcloud(tagged_words):
    tag_freq = nltk.FreqDist(tag for (word, tag) in tagged_words)
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(tag_freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)  # Display the figure in Streamlit

# Function to capture the output of upenn_tagset
def get_pos_tag_descriptions():
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        nltk.help.upenn_tagset()
        return buf.getvalue()

# Streamlit app
def main():
    st.title("POS Tagging and Word Cloud Generation")

    # User input
    user_input = st.text_area("Enter your statement here", "")

    # Checkbox to show POS tag descriptions
    if st.checkbox("Show list of POS tags"):
        pos_tag_descriptions = get_pos_tag_descriptions()
        st.text(pos_tag_descriptions)  # Display the POS tag descriptions

    if st.button("Analyze"):
        # Perform POS tagging
        tagged = pos_tagging(user_input)
        st.write("POS Tags:", tagged)

        # Generate word cloud
        generate_wordcloud(tagged)

if __name__ == "__main__":
    main()
