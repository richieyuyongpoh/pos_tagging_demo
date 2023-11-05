import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import contextlib
import io

# Download the necessary NLTK models, corpora, and lists
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Function to perform POS tagging
def pos_tagging(text):
    nltk_tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(nltk_tokens)
    return tagged

# Function to capture the output of upenn_tagset
def get_pos_tag_descriptions():
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        nltk.help.upenn_tagset()
        return buf.getvalue()

# Function to generate a bar graph of POS tag frequencies
def generate_pos_tag_graph(tagged_words):
    tags = [tag for word, tag in tagged_words]
    tag_freq = nltk.FreqDist(tags)
    tag_freq_df = pd.DataFrame.from_dict(tag_freq, orient='index', columns=['Frequency'])
    tag_freq_df.sort_values(by='Frequency', ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    tag_freq_df.plot(kind='bar', ax=ax)
    ax.set_title('Frequency Distribution of POS Tags')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('POS Tags')
    st.pyplot(fig)  # Display the figure in Streamlit

# Function to generate a word cloud from the actual words
def generate_wordcloud(words):
    word_freq = nltk.FreqDist(words)
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)  # Display the figure in Streamlit

# Streamlit app
def main():
    st.title("NLP Visualization")

    # Checkbox to show POS tag descriptions
    if st.checkbox("Show list of POS tags"):
        pos_tag_descriptions = get_pos_tag_descriptions()
        st.text(pos_tag_descriptions)  # Display the POS tag descriptions

    
    # User input
    user_input = st.text_area("Enter your statement here", "")

    if st.button("Analyze"):
        # Tokenize the user input and perform POS tagging
        nltk_tokens = nltk.word_tokenize(user_input)
        tagged = pos_tagging(user_input)

        # Display original POS tags and word cloud
        st.write("Original POS Tags:", tagged)
        generate_pos_tag_graph(tagged)
        generate_wordcloud([word for word, tag in tagged])

        # Process text: remove stopwords and apply stemming
        stop_words = set(stopwords.words('english'))
        porter = PorterStemmer()
        filtered_tokens = [porter.stem(word) for word in nltk_tokens if word.lower() not in stop_words]
        filtered_tagged = pos_tagging(' '.join(filtered_tokens))

        # Display processed POS tags and word cloud
        st.write("Processed POS Tags:", filtered_tagged)
        generate_pos_tag_graph(filtered_tagged)
        generate_wordcloud(filtered_tokens)

if __name__ == "__main__":
    main()
