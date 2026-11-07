from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "context_aware.joblib"

st.set_page_config(page_title="Context-Aware Humor Detection", page_icon="😂", layout="centered")
st.title("😂 Context-Aware Humor Detection")
st.caption("NLP mini-project · Context + utterance classification")

st.write(
    "Enter the situation or previous conversation, then the sentence you want to analyze. "
    "The model predicts whether the target utterance resembles humorous or non-humorous examples."
)

if not MODEL_PATH.exists():
    st.warning("Model not trained yet. In the project folder, run `python train.py`, then refresh this page.")
    st.stop()

model = joblib.load(MODEL_PATH)
context = st.text_area(
    "Conversation context",
    placeholder="Example: My friend studied all night but still forgot to bring a pen.",
    height=110
)
utterance = st.text_area(
    "Target utterance",
    placeholder="Example: Truly, a master of preparation.",
    height=90
)

if st.button("Detect humor", type="primary", use_container_width=True):
    if not utterance.strip():
        st.error("Please enter a target utterance.")
    else:
        combined = (
        f"CONTEXT: {context.strip()} "
        f"TARGET UTTERANCE: {utterance.strip()}"
    )

        prediction = model.predict([combined])[0]
        probabilities = model.predict_proba([combined])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)])

        if prediction == "humorous":
            st.success("Prediction: HUMOROUS 😂")
        else:
            st.info("Prediction: NON-HUMOROUS 🙂")
        st.metric("Model confidence", f"{confidence:.1%}")
        st.caption(
            "Confidence is the model's estimated class probability, not a guarantee. "
            "Humor is subjective and the demo model's reliability depends on the training data."
        )

with st.expander("Try an example"):
    st.write("**Context:** My friend said he would start studying at 6 AM, but woke up at noon.")
    st.write("**Utterance:** His time management is truly legendary.")
    st.write("This is an illustrative example; predictions may vary.")
