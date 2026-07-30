
import streamlit as st
import joblib
import pandas as pd
import re

# Load models
nb_model = joblib.load("naive_bayes_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
svm_model = joblib.load("linear_svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="SMS Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SMS Fraud Detection System")
st.write("Detect fraudulent and legitimate SMS messages using Machine Learning.")
st.caption("Version 1.0 | Powered by Machine Learning")

with st.form("sms_form"):

    user_message = st.text_area(
        "Enter your SMS message:",
         height=150,
         placeholder="Type or paste an SMS here..."
       )

    submitted = st.form_submit_button("🔍 Analyse SMS")


spam_keywords = {
      "Urgency": [
           "urgent", "immediately", "now", "today", "expire",
           "expired", "limited", "last chance"
      ],

       "Money": [
           "free", "gift", "cash", "reward", "bonus",
           "payment", "loan", "credit", "bank", "₦", "$"
      ],

       "Account": [
            "verify", "update", "login", "confirm",
            "password", "otp", "security", "account"
      ],

       "Action": [
            "click", "tap", "visit", "link", "claim"
      ]
}


def explain_message(message):

     message = message.lower()

     reasons = []

     for category, words in spam_keywords.items():
          found = []

          for word in words:
               if word.lower() in message:
                 found.append(word)

          if found:
               reasons.append((category, found))

     return reasons



def final_decision(nb_pred, lr_pred, svm_pred):
      votes = [nb_pred, lr_pred, svm_pred]

      spam_votes = votes.count(1)
      safe_votes = votes.count(0)

      if spam_votes >= 2:
           decision = "🚨 FRAUD SMS"
           reason = f"{spam_votes} out of 3 models classified this message as Fraud."
      else:
           decision = "✅ SAFE SMS"
           reason = f"{safe_votes} out of 3 models classified this message as Safe."

      return decision, reason


if submitted:

     if user_message.strip() == "":
          st.warning("Please enter an SMS message.")
     else:

          # Convert message to TF-IDF
          message_vector = tfidf.transform([user_message])

          # Predictions
          nb_pred = nb_model.predict(message_vector)[0]
          lr_pred = lr_model.predict(message_vector)[0]
          
          try:
              svm_pred = svm_model.predict(message_vector)[0]
          except Exception as e:
              st.error(f"SVM Error: {e}")
              svm_pred = lr_pred

          final_result, final_reason = final_decision(
                nb_pred,
                lr_pred,
                svm_pred
           )

          reasons = explain_message(user_message)

          # Confidence Scores
          nb_prob = nb_model.predict_proba(message_vector)[0]
          lr_prob = lr_model.predict_proba(message_vector)[0]


          # Display result

          st.subheader("🤖 Model Predictions")

          col1, col2, col3 = st.columns(3)

          with col1:
               st.info("Naive Bayes")
               st.write("Prediction:", "🚨 Fraud SMS" if nb_pred == 1 else "✅ Safe SMS")
               st.progress(float(max(nb_prob)))
               st.write(f"Confidence: {max(nb_prob)*100:.2f}%")

          with col2:
               st.info("Logistic Regression")
               st.write("Prediction:", "🚨 Fraud SMS" if lr_pred == 1 else "✅ Safe SMS")
               st.progress(float(max(lr_prob)))
               st.write(f"Confidence: {max(lr_prob)*100:.2f}%")

          with col3:
               st.info("Linear SVM")
               st.write("Prediction:", "🚨 Fraud SMS" if svm_pred == 1 else "✅ Safe SMS")

          # ==========================
          # URL Security Analysis
          # ==========================

          st.write("---")

          urls = re.findall(r'(https?://\S+|www\.\S+)', user_message)

          if urls:

               st.subheader("🔗 URL Security Check")

               suspicious = False

               for url in urls:

                    # Hyphenated domains
                    if "-" in url:
                        st.warning(f"⚠️ Suspicious domain (contains hyphen): {url}")
                        suspicious = True

                    # URL shorteners
                    elif any(short in url.lower() for short in [
                          "bit.ly",
                          "tinyurl",
                          "t.co",
                          "goo.gl",
                          "is.gd",
                          "ow.ly",
                          "rb.gy"
                       ]):
                          st.warning(f"⚠️ Shortened URL detected: {url}")
                          suspicious = True

                    # IP address instead of domain
                    elif re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", url):
                          st.warning(f"⚠️ URL uses an IP address instead of a domain: {url}")
                          suspicious = True

                    else:
                          st.success(f"✅ URL appears normal: {url}")
                # Multiple links
               if len(urls) > 1:
                    st.warning(f"⚠️ This SMS contains {len(urls)} links.")

                # Overall advice
               st.info(
                    "⚠️ URL analysis is rule-based and provides additional guidance. "
                    "The machine learning models make the final spam prediction."
               )


          st.write("---")

          st.subheader("🔍 Why was this message flagged?")

          if reasons:

               for category, words in reasons:
                      st.write(f"**{category}:** {', '.join(words)}")

          else:
               st.success("No suspicious keywords detected.")

          st.write("---")

          st.subheader("🏆 Final Decision")

          if "FRAUD" in final_result:
              st.error(final_result)
          else:
              st.success(final_result)

          st.info(final_reason)


st.write("---")

st.subheader("📂 Batch SMS Analysis")

uploaded_file = st.file_uploader("Upload a CSV file containing SMS messages",type=["csv"])


if uploaded_file is not None:

      df_batch = pd.read_csv(uploaded_file)

      st.write("### Uploaded Data")
      st.dataframe(df_batch.head())

      if "Message" not in df_batch.columns:
           st.error("CSV must contain a column named 'Message'.")

      else:

           vectors = tfidf.transform(df_batch["Message"])

           df_batch["Naive Bayes"] = nb_model.predict(vectors)
           df_batch["Logistic Regression"] = lr_model.predict(vectors)
           
           try:
              df_batch["Linear SVM"] = svm_model.predict(vectors)
           except Exception as e:
              st.error(f"SVM Error: {e}")
              df_batch["Linear SVM"] = df_batch["Logistic Regression"]

           def majority_vote(row):
                votes = [
                row["Naive Bayes"],
                row["Logistic Regression"],
                row["Linear SVM"]
                ]

                if votes.count(1) >= 2:
                     return "🚨 FRAUD SMS"
                else:
                     return "✅ SAFE SMS"

           df_batch["Final Decision"] = df_batch.apply(
           majority_vote,
           axis=1
           )

           st.write("### Prediction Results")

           st.dataframe(
                df_batch[
                    [
                        "Message",
                        "Naive Bayes",
                        "Logistic Regression",
                        "Linear SVM",
                        "Final Decision"
                    ]
                ]
           )


           csv = df_batch.to_csv(index=False).encode("utf-8")

           st.download_button(
               "📥 Download Results", csv,
               "sms_predictions.csv",
                "text/csv"
            )

st.write("---")

with st.expander("ℹ️ About this Project"):

    st.markdown("""
    ## 🛡️ SMS Fraud Detection System

    This application uses Machine Learning to detect whether an SMS message is fraudulent or legitimate.

    ### 🎯 Project Objective

    To help individuals identify suspicious SMS messages before clicking malicious links or revealing sensitive information.

    ### 🤖 Machine Learning Models

    - Naive Bayes
    - Logistic Regression
    - Linear Support Vector Machine (SVM)

    A majority voting system combines the predictions of all three models to produce the final decision.

    ### 🔒 Features

    ✅ Real-time SMS analysis

    ✅ Confidence scores

    ✅ Keyword explainability

    ✅ URL Security Check

    ✅ Batch CSV analysis

    ✅ Download prediction results

    ### 🛠 Technologies Used

    - Python
    - Scikit-learn
    - Pandas
    - TF-IDF Vectorizer
    - Streamlit

    ### 👨‍💻 Developer

    **Aderoju Adam Damilare**

    Founder of AdaD Group and InspireWithAdaD.

    Passionate about building practical Artificial Intelligence and Cybersecurity solutions that help people stay safe online, make smarter decisions, and leverage technology to solve real-world problems.

    ### 🌐 Connect with Me

    📧 Email: inspirewithadad@gmail.com

    💼 LinkedIn: https://www.linkedin.com/in/inspirewithadad

    📘 Facebook: InspireWithAdaD

    📸 Instagram: @InspireWithAdaD

    ▶️ YouTube: @InspireWithAdaD

    💻 GitHub: https://github.com/inspirewithadad

    ### 📊 Version

    Version 1.0

    July 2026
    """)

st.write("---")
st.caption("🛡️ SMS Fraud Detection System | Version 1.0 | Powered by Machine Learning")






