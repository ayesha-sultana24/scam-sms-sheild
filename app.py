import joblib
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Scam SMS Shield | AI Fraud Defense Portal",
    page_icon="🛡️",
    layout="centered",
)

# Load model and components
model = joblib.load("model.joblib")
vec, clf = model.steps[0][1], model.steps[1][1]

# Enhanced Scam Taxonomy with Independent Verification Playbooks
TYPES = {
    "Fake KYC / Bank Block": {
        "words": ["kyc", "blocked", "account", "pan"],
        "tip": (
            "Banks NEVER ask for KYC updates or passwords through SMS links."
        ),
        "steps": [
            "Open your bank's OFFICIAL app directly (do not tap any link).",
            (
                "Call the customer service number printed on the back of your"
                " bank debit card."
            ),
            "Visit your local bank branch to confirm your account status.",
        ],
    },
    "UPI / Refund Trap": {
        "words": ["upi", "refund", "pin", "cashback"],
        "tip": (
            "Golden Rule: You NEVER enter a UPI PIN to receive money or"
            " refunds."
        ),
        "steps": [
            "Check your bank balance directly inside Google Pay / PhonePe.",
            "If someone sent a request, REJECT it immediately.",
            "Never share OTPs or accept collect requests from strangers.",
        ],
    },
    "Electricity / Utility Cut Threat": {
        "words": ["electricity", "disconnected", "power", "bill"],
        "tip": (
            "Electricity boards do NOT send personal mobile numbers for bill"
            " payments."
        ),
        "steps": [
            "Log in to your official state electricity portal or payment app.",
            "Check your actual bill status under the Utility tab in UPI apps.",
            (
                "Ignore threats of immediate disconnection within hours via"
                " SMS."
            ),
        ],
    },
    "Courier / Customs Duty Trap": {
        "words": ["parcel", "courier", "customs", "delivery"],
        "tip": (
            "Customs and postal services never demand payments via personal"
            " UPI handles."
        ),
        "steps": [
            "Go to the official website (e.g., India Post / BlueDart).",
            "Manually enter your tracking code on the official site.",
            "Refuse delivery if the sender demands unexpected cash on delivery.",
        ],
    },
    "Lottery / Job Scam": {
        "words": [
            "lottery",
            "prize",
            "won",
            "winner",
            "job",
            "earn",
            "registration",
        ],
        "tip": (
            "Genuine employers and contests never ask for upfront payments or"
            " registration fees."
        ),
        "steps": [
            "Verify the company on LinkedIn or official career pages.",
            "Never pay money for receiving a job offer or prize.",
        ],
    },
}


def explain(msg, top=4):
    """Return the words that pushed this message toward 'scam'."""
    c = vec.transform([msg]).multiply(clf.coef_[0]).tocoo()
    pairs = sorted(zip(c.data, c.col), reverse=True)[:top]
    names = vec.get_feature_names_out()
    return [names[j] for v, j in pairs if v > 0]


# --- APP HEADER ---
st.title("🛡️ Scam SMS Shield")
st.caption(
    "**AI Fraud Defense & Verification Portal** | Aligned with UN SDG 16"
)

# --- INPUT SECTION ---
st.markdown("### 📩 Paste Suspicious SMS Below")
msg = st.text_area(
    "SMS Message Text:",
    placeholder=(
        "E.g. Dear customer your SBI account is blocked. Update KYC now at"
        " http://bit.ly/123"
    ),
    height=100,
)

if st.button("🛡️ Analyze & Verify Message", type="primary") and msg.strip():
    p = model.predict_proba([msg])[0][1]

    st.markdown("---")
    st.subheader("📊 Analysis Summary")

    # --- VERDICT DISPLAY ---
    if p >= 0.5:
        st.error(f"🚨 **HIGH RISK DETECTED** (Risk Score: {p * 100:.1f}%)")

        # Risk Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Threat Probability", f"{p * 100:.1f}%")
        m2.metric("Threat Level", "CRITICAL" if p > 0.8 else "MODERATE")
        m3.metric("Action Required", "DO NOT CLICK LINKS")

        # Explainable AI Output
        flagged_words = explain(msg)
        if flagged_words:
            st.warning(
                "🔑 **Why was this flagged? (Explainable AI Trigger Words):** "
                + ", ".join([f"`{w}`" for w in flagged_words])
            )

        # Matched Category & Action Plan
        low = msg.lower()
        matched_data = None
        matched_name = ""

        for name, data in TYPES.items():
            if any(w in low for w in data["words"]):
                matched_data = data
                matched_name = name
                break

        st.markdown("---")
        st.subheader("🔍 Independent Safe Verification Steps")

        if matched_data:
            st.info(
                f"⚠️ **Detected Pattern:** {matched_name}\n\n💡"
                f" **Core Safety Rule:** {matched_data['tip']}"
            )
            st.markdown("**Follow these steps to safely verify without risk:**")
            for step in matched_data["steps"]:
                st.markdown(f"- ✅ {step}")
        else:
            st.info(
                "⚠️ **Unclassified Risk Pattern Detected.** Avoid interacting"
                " with links or unverified numbers."
            )
            st.markdown("**General Safe Verification Steps:**")
            st.markdown(
                "- ✅ Check official sender IDs (e.g., `AX-HDFCBK` vs regular"
                " 10-digit mobile numbers)."
            )
            st.markdown(
                "- ✅ Search the official helpline number of the organization"
                " on Google."
            )

        # Action Protocols
        st.markdown("---")
        st.subheader("🚨 Immediate Action Protocol")
        st.markdown(
            "1. **Clicked a link?** Disconnect internet immediately and change"
            " banking passwords."
        )
        st.markdown(
            "2. **Lost money?** Call **1930** (National Cybercrime Helpline)"
            " within the first 1-2 hours (Golden Hour)."
        )
        st.markdown(
            "3. **Report Cybercrime:** Submit details at"
            " [cybercrime.gov.in](https://cybercrime.gov.in)."
        )

    else:
        st.success(f"✅ **LIKELY SAFE** (Safety Score: {(1 - p) * 100:.1f}%)")
        st.markdown("This message does not contain common financial fraud indicators.")
        st.markdown("**Good Practices:** Always ensure official bank SMS come from official headers (like `VK-SBIIN` or `AD-HDFCBK`).")

    # Developer / Viva Inspector
    with st.expander("🛠️ Technical Model Inspection (For Evaluators)"):
        st.json({
            "algorithm": "Logistic Regression with TF-IDF Vectorization",
            "scam_probability": float(p),
            "safe_probability": float(1 - p),
            "feature_coefficients": explain(msg, top=6),
        })
