
import streamlit as st
import joblib
import pandas as pd
import re

# Load models
nb_model = joblib.load("naive_bayes_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
sgd_model = joblib.load("sgd_classifier_model.pkl")
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

      "Urgency": ["urgent", "immediately", "now", "today", "expire", "expired", "limited", "last chance", "blocked", "restricted", "suspended", "avoid", "failure"],

      "Account & Security": ["verify", "update", "confirm", "login", "signin", "password", "otp", "pin", "atm", "security", "account", "bvn", "nin"],

      "Banking & Finance": ["bank", "credit", "debit", "transfer", "withdraw", "transaction", "payment", "loan", "wallet", "cash", "money", "reward", "bonus", "gift", "free", "₦", "$"],

      "Banks & FinTech": ["gtbank", "access", "zenith", "uba", "fcmb", "sterling", "wema", "opay", "palmpay", "moniepoint", "kuda"],

      "Telecom": ["mtn", "airtel", "glo", "9mobile", "airtime", "bundle", "data", "sim"],

      "Government & Grants": ["grant", "empowerment", "nirsal", "cbn", "cac", "government", "federal", "fg"],

      "Prize & Lottery": ["winner", "won", "prize", "jackpot", "million", "inheritance"],

      "Action Required": ["click", "tap", "visit", "claim", "reply", "download", "activate", "register"],

      "Delivery": ["package", "delivery", "dispatch", "tracking", "courier", "dhl", "fedex"],

      "Cryptocurrency": ["bitcoin", "btc", "crypto", "usdt", "wallet"]
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


def final_decision(lr_pred):

      if lr_pred == 1:
              return "🚨 Likely Fraudulent SMS"

      return "✅ Likely Safe SMS"



if submitted:

     if user_message.strip() == "":
          st.warning("Please enter an SMS message.")
     else:

          # Convert message to TF-IDF
          message_vector = tfidf.transform([user_message])

          # Predictions
          nb_pred = nb_model.predict(message_vector)[0]
          lr_pred = lr_model.predict(message_vector)[0]
          sgd_pred = sgd_model.predict(message_vector)[0]

          final_result = final_decision(lr_pred)

          reasons = explain_message(user_message)

          # Confidence Scores
          nb_prob = nb_model.predict_proba(message_vector)[0]
          lr_prob = lr_model.predict_proba(message_vector)[0]
          sgd_prob = sgd_model.predict_proba(message_vector)[0]


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
               st.info("SGD Classifier")
               st.write("Prediction:", "🚨 Fraud SMS" if sgd_pred == 1 else "✅ Safe SMS")
               st.progress(float(max(sgd_prob)))
               st.write(f"Confidence: {max(sgd_prob)*100:.2f}%")

          # ==========================
          # URL Security Analysis
          # ==========================

          st.write("---")

          urls = re.findall(r'(https?://\S+|www\.\S+)', user_message)

          if urls:

               st.subheader("🔗 URL Security Check")


               for url in urls:

                    # Hyphenated domains
                    if "-" in url:
                        st.warning(f"⚠️ Suspicious domain (contains hyphen): {url}")

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


                    # IP address instead of domain
                    elif re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", url):
                          st.warning(f"⚠️ URL uses an IP address instead of a domain: {url}")

                    else:
                          st.success(f"✅ No obvious risk indicators detected in this URL: {url}")
                # Multiple links
               if len(urls) > 1:
                    st.warning(f"⚠️ This SMS contains {len(urls)} links.")

                # Overall advice
               st.info(
                    "⚠️ URL analysis is rule-based and provides additional guidance. "
                    "The Machine Learning models generate the final prediction. URL analysis provides additional security guidance."
               )


          st.write("---")

          st.subheader("🔍 Why was this message flagged?")

          if reasons:

               for category, words in reasons:
                      st.write(f"**{category}:** {', '.join(words)}")

          else:
               st.success("No suspicious keywords detected.")


          st.write("---")

          st.subheader("🏆 Final Prediction")

          if lr_pred == 1:

              st.error(final_result)
              st.write(f"**Confidence:** {max(lr_prob)*100:.2f}%")


              st.error("""
              ### 🚨 Security Recommendation

              This message has been identified as **Likely Fraudulent** by the Logistic Regression model.

              Please consider the following before taking any action:

              • Do NOT click any links.

              • Do NOT share your BVN, ATM PIN, OTP, password, or personal information.

              • Do NOT send money or respond immediately.

              • Verify the message directly through the official website or customer service of the organisation.
              
              • If you suspect fraud, report the message to your telecom provider or bank.

              ⚠️ Although this prediction has a high confidence score, no AI model is 100% accurate. Always use your own judgement.
              """)
          else:

                st.success(final_result)
                st.write(f"**Confidence:** {max(lr_prob)*100:.2f}%")


                st.warning("""
                ### 🛡️ Security Advisory

                This message has been identified as **Likely Safe** by the Logistic Regression model.

                However, no Machine Learning model is 100% accurate.

                Please remain cautious if the message:

                • Requests your BVN, ATM PIN, OTP, or password.

                • Contains suspicious links or unfamiliar websites.

                • Creates urgency ("Immediately", "Act now", "Account suspended").

                • Promises prizes, grants, loans, or unexpected rewards.

                • Requests personal or banking information.

                When in doubt, always verify directly with the official organisation before taking any action.
                """)


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
           df_batch["SGD Classifier"] = sgd_model.predict(vectors)

           # Final decision comes directly from Logistic Regression
           df_batch["Final Decision"] = df_batch["Logistic Regression"].apply(
                lambda x: "🚨 Likely Fraudulent SMS" if x == 1 else "✅ Likely Safe SMS"
                )

           df_batch["Recommendation"] = df_batch["Logistic Regression"].apply(
                lambda x:
                "Verify sender before taking action." if x == 1
                else "🛡️ No major fraud indicators detected. Always verify unexpected requests."
               )


           st.write("### Prediction Results")

           st.dataframe(
                df_batch[
                    [
                        "Message",
                        "Naive Bayes",
                        "Logistic Regression",
                        "SGD Classifier",
                        "Final Decision",
                        "Recommendation"
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

    - Naive Bayes Classifier
    - Logistic Regression (Primary Prediction Model)
    - Stochastic Gradient Descent (SGD) Classifier

    Logistic Regression serves as the primary prediction engine because it achieved the highest performance during model evaluation. The Naive Bayes and the Stochastic Gradient Descent (SGD) Classifier are displayed alongside it for comparison and performance analysis.

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






