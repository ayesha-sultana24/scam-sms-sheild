import joblib, streamlit as st

model = joblib.load("model.joblib")
vec, clf = model.steps[0][1], model.steps[1][1]

# Simple keyword rules to label the scam type and show a safe-advice tip
TYPES = {
  "Fake KYC / bank block": (["kyc", "blocked", "account", "pan"], "Banks never ask for KYC updates through SMS links. Visit your branch or the official app."),
  "UPI / refund trap": (["upi", "refund", "pin", "cashback"], "You never need to enter your UPI PIN to RECEIVE money."),
  "Lottery / prize": (["lottery", "prize", "won", "winner"], "Real prizes don't ask you to pay a fee first."),
  "Job offer": (["job", "earn", "registration", "daily"], "Genuine employers don't charge registration fees."),
  "Courier / customs fee": (["parcel", "courier", "customs", "delivery"], "Check the courier's official app or website, not the SMS link."),
  "Electricity cut threat": (["electricity", "disconnected", "power"], "Check your bill on the official electricity board site."),
}

def explain(msg, top=4):
    c = vec.transform([msg]).multiply(clf.coef_[0]).tocoo()
    pairs = sorted(zip(c.data, c.col), reverse=True)[:top]
    names = vec.get_feature_names_out()
    return [names[j] for v, j in pairs if v > 0]

st.title("🛡️ Scam SMS Shield")
st.caption("Paste a suspicious SMS to check it. SDG 16: safer communities, less fraud.")
msg = st.text_area("SMS text")

if st.button("Check") and msg.strip():
    p = model.predict_proba([msg])[0][1]
    if p >= 0.5:
        st.error(f"⚠️ Likely SCAM ({p:.0%} confidence)")
        st.write("Flagged because of:", ", ".join(explain(msg)) or "overall pattern")
        low = msg.lower()
        for name, (words, tip) in TYPES.items():
            if any(w in low for w in words):
                st.warning(f"Looks like: {name}. {tip}")
                break
        st.info("Never share OTP/PIN. Report fraud: call 1930 or visit cybercrime.gov.in (please verify before final submission).")
    else:
        st.success(f"✅ Looks safe ({1-p:.0%} confidence)")
    st.caption("This is a student prototype and can make mistakes. When in doubt, don't click.")