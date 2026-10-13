import os
import joblib
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Scam SMS Shield | Fraud Defense Portal",
    layout="centered",
)

# Load Model Artifacts with Error Handling
@st.cache_resource
def load_model_pipeline():
    if not os.path.exists("model.joblib"):
        return None
    return joblib.load("model.joblib")

model = load_model_pipeline()

if model is None:
    st.error("System Error: 'model.joblib' file not found in the repository. Please verify your file upload.")
    st.stop()

# Safely extract vectorizer and classifier from scikit-learn pipeline
try:
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        vec = steps[0]
        clf = steps[1]
    else:
        vec = model.steps[0][1]
        clf = model.steps[1][1]
except Exception:
    # Fallback default indexing
    vec = model.steps[0][1]
    clf = model.steps[1][1]

# Institutional Taxonomies & Independent Verification Playbooks
THREAT_TYPES = {
    "Fake KYC / Bank Block": {
        "keywords": ["kyc", "blocked", "account", "pan", "update", "verify"],
        "advisory": "Financial institutions never request KYC updates, PINs, or credentials via direct SMS hyperlinks.",
        "verification_steps": [
            "Open your bank's official proprietary mobile application directly.",
            "Contact customer support using the verified telephone number printed on your debit card.",
            "Visit your designated local branch to review account restrictions.",
        ],
    },
    "UPI / Refund Trap": {
        "keywords": ["upi", "refund", "pin", "cashback", "receive", "reward"],
        "advisory": "A UPI PIN is exclusively required for authorizing outbound debit transactions, never for receiving funds.",
        "verification_steps": [
            "Inspect your ledger balance directly within your trusted payment gateway app.",
            "Reject and block all unsolicited collect requests or payment prompts.",
            "Never disclose your UPI PIN or virtual payment address credentials.",
        ],
    },
    "Utility Disconnection Threat": {
        "keywords": ["electricity", "disconnected", "power", "bill", "current", "eb"],
        "advisory": "State utility boards do not issue final service disconnection notices from personal mobile numbers.",
        "verification_steps": [
            "Access your state electricity board's official web portal.",
            "Verify your monthly billing status through official utility aggregator apps.",
            "Disregard urgent messaging demanding immediate mobile wallet transfers.",
        ],
    },
    "Courier / Customs Duty Trap": {
        "keywords": ["parcel", "courier", "customs", "delivery", "shipment"],
        "advisory": "Postal and customs authorities do not levy clearance fees via personal digital wallet handles.",
        "verification_steps": [
            "Navigate directly to the official logistics provider tracking portal.",
            "Manually input your tracking reference on the authenticated web domain.",
            "Refuse consignments demanding cash-on-delivery settlements via unverified channels.",
        ],
    },
    "Employment / Prize Fraud": {
        "keywords": ["lottery", "prize", "won", "winner", "job", "earn", "registration", "task"],
        "advisory": "Legitimate employers and official contests do not mandate upfront registration fees.",
        "verification_steps": [
            "Authenticate corporate entities via verified professional networks.",
            "Decline financial transfers requested as a prerequisite for employment or reward disbursement.",
        ],
    },
}

def explain(msg, top=4):
    """Extract top feature coefficients pushing the text toward malicious classification."""
    try:
        c = vec.transform([msg]).multiply(clf.coef_[0]).tocoo()
        pairs = sorted(zip(c.data, c.col), reverse=True)[:top]
        names = vec.get_feature_names_out()
        return [names[j] for v, j in pairs if v > 0]
    except Exception:
        return []

# --- UI LAYOUT ---

st.title("Scam SMS Shield")
st.markdown("**Enterprise Fraud Detection & Verification Portal** | Framework Aligned with UN SDG 16")
st.markdown("---")

st.subheader("Message Analysis Console")
sms_input = st.text_area(
    "SMS Content Input",
    placeholder="Paste the complete text message content for analysis...",
    height=120,
)

col_action, col_spacer = st.columns([1, 3])
with col_action:
    analyze_btn = st.button("Run Diagnostics", type="primary")

if analyze_btn and sms_input.strip():
    try:
        probability = model.predict_proba([sms_input])[0][1]
    except Exception as e:
        st.error(f"Prediction execution failed: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("Diagnostic Results")

    if probability >= 0.5:
        # High Risk Classification Output
        st.error(f"**Classification: HIGH RISK THREAT DETECTED** (Confidence: {probability * 100:.2f}%)")

        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Risk Probability", f"{probability * 100:.1f}%")
        metric_2.metric("Threat Category", "Malicious Smishing")
        metric_3.metric("Recommended Action", "Isolate & Block")

        drivers = explain(sms_input)
        if drivers:
            st.markdown("**Primary Linguistic Risk Indicators:**")
            st.code(", ".join(drivers), language=None)

        # Pattern Matching for Independent Verification Playbook
        text_lower = sms_input.lower()
        matched_profile = None
        for category, details in THREAT_TYPES.items():
            if any(kw in text_lower for kw in details["keywords"]):
                matched_profile = (category, details)
                break

        st.markdown("---")
        st.subheader("Independent Verification Protocol")

        if matched_profile:
            cat_name, cat_data = matched_profile
            st.info(f"**Identified Vector:** {cat_name}\n\n**Advisory:** {cat_data['advisory']}")
            st.markdown("**Mandatory Verification Steps:**")
            for step in cat_data["verification_steps"]:
                st.markdown(f"* {step}")
        else:
            st.info("**Vector Classification:** Unidentified Fraud Pattern. Treat with extreme caution.")
            st.markdown("**General Verification Steps:**")
            st.markdown("* Cross-reference sender identification strings with official organizational directories.")
            st.markdown("* Never interact with embedded shortened URIs (e.g., bit.ly, tinyurl).")

        st.markdown("---")
        st.subheader("Incident Response Protocol")
        st.markdown("1. **Credential Exposure:** If links were accessed, immediately terminate active sessions and update security keys.")
        st.markdown("2. **Financial Loss:** Report incidents to the National Cybercrime Helpline (**1930**) within the active response window.")
        st.markdown("3. **Formal Filing:** Register grievances via the official portal at [cybercrime.gov.in](https://cybercrime.gov.in).")

    else:
        # Benign Classification Output
        st.success(f"**Classification: LOW RISK / BENIGN** (Confidence: {(1 - probability) * 100:.2f}%)")
        st.markdown("The submitted text does not exhibit known statistical markers associated with financial fraud.")
        st.markdown("**Standard Hygiene:** Ensure official organizational communications originate from authenticated enterprise sender identifiers.")

    # Technical Model Telemetry Expander
    with st.expander("Model Telemetry & Feature Inspector"):
        st.json({
            "vectorizer": "TF-IDF (Unigram & Bigram Configuration)",
            "estimator": "Optimized Logistic Regression Classifier",
            "calculated_probability_malicious": float(probability),
            "calculated_probability_benign": float(1 - probability),
            "feature_coefficients": explain(sms_input, top=6),
        })

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.85em;'>Academic Capstone Project | Machine Learning & Applied AI</div>", unsafe_allow_html=True)
