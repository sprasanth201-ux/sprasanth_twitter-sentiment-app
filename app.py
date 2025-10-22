# ==========================================================
# 🧠 Twitter Sentiment Analysis Web Application
# Developer: Prasanth (Data Analyst Department)
# ==========================================================
# Features:
# 1️⃣ Accept multiple tweets with User IDs (format: id, tweet)
# 2️⃣ Predict Positive / Negative sentiments
# 3️⃣ Show table + bar chart + CSV download
# ==========================================================

import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 🔹 Load Pre-trained Model and Vectorizer
# ----------------------------------------------------------
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ----------------------------------------------------------
# 🔹 Function to Clean Tweet Text
# ----------------------------------------------------------
def clean_tweet(tweet):
    """Cleans tweet text by removing URLs, mentions, hashtags, special characters, and extra spaces."""
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)
    tweet = tweet.lower().strip()
    return tweet

# ----------------------------------------------------------
# 🔹 Streamlit UI Setup
# ----------------------------------------------------------
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")
st.title("📊 Twitter Sentiment Analysis Dashboard (With User ID)")
st.write("Enter multiple tweets with **User IDs**, or upload a CSV file for sentiment prediction.")

# ----------------------------------------------------------
# 🔹 User Input Section
# ----------------------------------------------------------
st.header("✍️ Enter Multiple Tweets with IDs (Format: ID, Tweet)")
example = "Example:\n123, I love this phone!\n456, I hate the battery life."
tweets_input = st.text_area("Paste or type your tweets below:", placeholder=example)

if st.button("Analyze Tweets"):
    if tweets_input.strip() != "":
        # Split input lines
        lines = [l.strip() for l in tweets_input.split('\n') if l.strip() != ""]
        user_ids, tweets = [], []

        for line in lines:
            parts = line.split(",", 1)
            if len(parts) == 2:
                user_ids.append(parts[0].strip())
                tweets.append(parts[1].strip())
            else:
                # If no ID found, mark as Unknown
                user_ids.append("Unknown")
                tweets.append(parts[0].strip())

        # Clean text
        clean_tweets = [clean_tweet(t) for t in tweets]
        X_vec = vectorizer.transform(clean_tweets)
        predictions = model.predict(X_vec)

        # Create DataFrame
        results = []
        for uid, tweet, pred in zip(user_ids, tweets, predictions):
            sentiment = "Positive" if pred == 4 or pred == "positive" else "Negative"
            results.append((uid, tweet, sentiment))

        df_results = pd.DataFrame(results, columns=["User ID", "Tweet", "Sentiment"])

        # Show results
        st.subheader("📋 Sentiment Predictions")
        st.dataframe(df_results, use_container_width=True)

        # Sentiment distribution visualization
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df_results['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red'])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Summary")
        st.pyplot(fig)

        # Download results
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=df_results.to_csv(index=False).encode('utf-8'),
            file_name="tweet_sentiment_with_userid.csv",
            mime="text/csv"
        )

    else:
        st.warning("⚠️ Please enter at least one tweet in the correct format (ID, Tweet).")

st.markdown("---")
st.caption("Developed by **Prasanth** | Department: Data Analyst | Zencorp Techno Solutions")
