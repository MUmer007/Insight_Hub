"""
2_Ticket_Classifier.py

A page in the InsightHub multi-page Streamlit app.
Lets a user paste in a support ticket description and see it classified
live, using the TF-IDF baseline model trained in Phase 3.
"""

from pathlib import Path

import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "ticket_classifier_baseline.joblib"

st.set_page_config(page_title="Ticket Classifier", page_icon="🎫")
st.title("🎫 Support Ticket Classifier")
st.caption("Paste a support ticket's subject and description to see it auto-categorized.")


@st.cache_resource
def load_ticket_model():
    return joblib.load(MODEL_PATH)


model = load_ticket_model()

subject = st.text_input("Ticket Subject", placeholder="e.g. Refund not received")
description = st.text_area(
    "Ticket Description",
    placeholder="e.g. I returned my order three weeks ago and still haven't received my refund...",
    height=150,
)

if st.button("Classify Ticket"):
    if not description.strip():
        st.warning("Please enter a ticket description first.")
    else:
        combined_text = f"{subject} {description}".strip()

        prediction = model.predict([combined_text])[0]
        probabilities = model.predict_proba([combined_text])[0]
        classes = model.classes_

        st.divider()
        st.subheader(f"Predicted category: **{prediction}**")

        # Show confidence for every category, sorted highest first
        prob_dict = dict(zip(classes, probabilities))
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        st.write("Confidence breakdown:")
        for category, prob in sorted_probs:
            st.progress(float(prob), text=f"{category}: {prob:.1%}")

        st.caption(
            "This is the TF-IDF + Logistic Regression baseline from Phase 3. "
            "A fine-tuned DistilBERT model was also trained for comparison (see MLflow for metrics)."
        )