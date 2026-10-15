"""Scam SMS Shield: a plain-language scam checker for SMS messages (SDG 16).

Two layers decide the verdict:
  1. a text model (TF-IDF + logistic regression) that scores the wording
  2. eight rule-based warning signs that describe what scammers actually do
The page shows both, so every verdict comes with reasons a person can check.
"""
import html
import re

import joblib
import streamlit as st

AUTHOR = "Ayesha"  # shown in the footer; change if you like

st.set_page_config(page_title="Scam SMS Shield", layout="wide", initial_sidebar_state="collapsed")

# Fill these in with your own Colab results (precision, recall, F1 as 0-1 values).
# Rows left as None are hidden from the page.
METRICS = [
    ("Public data only", None, None, None),
    ("With Indian examples added", None, None, None),
]

# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&display=swap');

@property --a { syntax:'<angle>'; inherits:false; initial-value:0deg; }

:root {
  --bg:#F2F4F7; --surface:#FFFFFF; --ink:#101C2C; --body:#2B394C; --muted:#5A6779; --line:#DCE1E8; --soft:#F7F9FB;
  --primary:#17365A; --primary-2:#0F2744;
  --scam:#B42318; --scam-bg:#FDEDEB; --scam-line:#F2BDB7; --scam-ink:#5E130C;
  --warn:#8A5A00; --warn-bg:#FFF5DB; --warn-line:#EBCB7C; --warn-ink:#4A3000;
  --safe:#14633F; --safe-bg:#E6F4EC; --safe-line:#A9D8BE; --safe-ink:#0B3A25;
  --display:'Bricolage Grotesque','Atkinson Hyperlegible',system-ui,sans-serif;
  --text:'Atkinson Hyperlegible',system-ui,-apple-system,'Segoe UI',sans-serif;
}

/* Force one light look everywhere, even if the visitor's device is in dark mode */
.stApp { color-scheme:light; background:var(--bg); color:var(--ink); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp textarea, .stApp input,
.stApp summary, .stApp td, .stApp th, .stApp div, .stApp h1, .stApp h2, .stApp h3 { font-family:var(--text); }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility:hidden; height:0; }
.block-container { max-width:1180px; padding:1.4rem 1.5rem 3rem; }
.stApp p, .stApp li { color:var(--body); }

/* ---- masthead */
.masthead { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding-bottom:.9rem; border-bottom:1px solid var(--line); }
.brand { display:flex; align-items:center; gap:.7rem; }
.crest { width:30px; height:34px; background:var(--primary); clip-path:polygon(50% 0,100% 14%,100% 56%,50% 100%,0 56%,0 14%); position:relative; }
.crest::after { content:""; position:absolute; inset:7px 7px 11px; background:#fff; clip-path:polygon(50% 0,100% 16%,100% 56%,50% 100%,0 56%,0 16%); }
.crest::before { content:""; position:absolute; left:11px; top:11px; width:8px; height:8px; background:var(--scam); border-radius:50%; z-index:2; }
.wordmark { font-family:var(--display); font-weight:700; font-size:1.3rem; color:var(--ink); letter-spacing:-.01em; }
.sdg { font-size:.92rem; color:var(--primary); background:#E4EBF4; border-radius:999px; padding:.3rem .85rem; font-weight:700; white-space:nowrap; }

.headline { font-family:var(--display); font-size:3rem; line-height:1.05; font-weight:700; margin:1.8rem 0 .5rem; letter-spacing:-.025em; color:var(--ink); }
.lede { font-size:1.15rem; line-height:1.55; color:var(--muted); max-width:42rem; margin:0 0 1.2rem; }

/* ---- nav (Streamlit tabs, always visible) */
.stTabs [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding:.75rem 1rem; background:transparent; height:auto; }
.stTabs [data-baseweb="tab"] p { color:var(--muted) !important; font-size:1.05rem; font-weight:400; }
.stTabs [data-baseweb="tab"]:hover p { color:var(--ink) !important; }
.stTabs [aria-selected="true"] p { color:var(--ink) !important; font-weight:700; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--scam); height:3px; }
.stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1.4rem; }

/* ---- panels */
.st-key-input, .st-key-result { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:1.4rem 1.5rem 1.5rem; box-shadow:0 1px 2px rgba(16,28,44,.05); }
.panel-title { font-family:var(--display); font-size:1.25rem; font-weight:700; color:var(--ink); margin:0 0 .7rem; }
.small-label { font-size:.95rem; color:var(--muted); margin:1rem 0 .4rem; }

[data-testid="stTextArea"] textarea { background:var(--soft) !important; color:var(--ink) !important; border:2px solid var(--line); border-radius:10px; font-size:1.08rem; line-height:1.55; padding:.9rem 1rem; }
[data-testid="stTextArea"] textarea::placeholder { color:#7B8797; opacity:1; }
[data-testid="stTextArea"] textarea:focus { border-color:var(--primary); box-shadow:0 0 0 3px rgba(23,54,90,.16); background:#fff !important; }
[data-testid="stTextArea"] [data-baseweb="textarea"], [data-testid="stTextArea"] [data-baseweb="base-input"] { background:transparent; border:none; }
[data-testid="stTextAreaRootElement"] + div, [data-testid="InputInstructions"] { color:var(--muted) !important; }

/* example chips */
.st-key-picker button, .st-key-picker [role="option"], .st-key-picker [data-baseweb="button-group"] button {
  background:#fff !important; color:var(--ink) !important; border:1px solid var(--line) !important; border-radius:999px !important; font-size:.93rem; padding:.3rem .85rem; min-height:2.1rem; }
.st-key-picker button:hover { border-color:var(--primary) !important; background:#EEF3F9 !important; }
.st-key-picker button p, .st-key-picker button div { color:var(--ink) !important; font-size:.93rem; }

/* buttons */
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:700; font-size:1.02rem; min-height:3rem; padding:.55rem 1.2rem; transition:background .15s, border-color .15s; }
.stButton > button p, .stDownloadButton > button p { font-weight:700; }
.stButton > button[data-testid="stBaseButton-primary"] { background:var(--primary); border:2px solid var(--primary); }
.stButton > button[data-testid="stBaseButton-primary"] p { color:#fff !important; }
.stButton > button[data-testid="stBaseButton-primary"]:hover { background:var(--primary-2); border-color:var(--primary-2); }
.stButton > button[data-testid="stBaseButton-secondary"], .stDownloadButton > button { background:#fff; border:2px solid var(--line); }
.stButton > button[data-testid="stBaseButton-secondary"] p, .stDownloadButton > button p { color:var(--ink) !important; }
.stButton > button[data-testid="stBaseButton-secondary"]:hover, .stDownloadButton > button:hover { border-color:var(--ink); background:#fff; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline:3px solid var(--primary); outline-offset:2px; }
.notice { color:var(--scam); font-weight:700; margin:.5rem 0 0; }

/* ---- verdict */
.verdict { display:flex; align-items:center; justify-content:space-between; gap:1.2rem; padding:1.3rem 1.5rem; border-radius:14px; border:1px solid; flex-wrap:wrap; }
.verdict .vtext { flex:1 1 15rem; }
.verdict h2 { font-family:var(--display); font-size:1.95rem; line-height:1.12; margin:0 0 .4rem; font-weight:700; color:inherit; letter-spacing:-.015em; }
.verdict p { margin:0; font-size:1.08rem; line-height:1.5; color:inherit; }
.v-scam { background:var(--scam-bg); border-color:var(--scam-line); color:var(--scam-ink); --c:var(--scam); }
.v-suspicious { background:var(--warn-bg); border-color:var(--warn-line); color:var(--warn-ink); --c:#C98A00; }
.v-safe { background:var(--safe-bg); border-color:var(--safe-line); color:var(--safe-ink); --c:var(--safe); }

/* the dial: the one memorable element */
.dial { position:relative; width:200px; height:104px; overflow:hidden; flex:0 0 auto; margin:0 auto; }
.dial::before { content:""; position:absolute; left:0; top:0; width:200px; height:200px; border-radius:50%;
  background:conic-gradient(from 270deg, var(--c) 0deg var(--a), rgba(16,28,44,.12) var(--a) 180deg, transparent 180deg);
  -webkit-mask:radial-gradient(circle closest-side, transparent 70%, #000 71%); mask:radial-gradient(circle closest-side, transparent 70%, #000 71%);
  animation:sweep 1.1s cubic-bezier(.2,.7,.2,1) both; }
.dial .num { position:absolute; left:0; right:0; bottom:0; text-align:center; line-height:1; }
.dial .num b { display:block; font-family:var(--display); font-size:2.5rem; margin-bottom:.15rem; font-weight:700; color:inherit; }
.dial .num span { font-size:.85rem; opacity:.8; display:block; }
@keyframes sweep { from { --a:0deg; } to { --a:var(--t); } }

.tiles { display:grid; grid-template-columns:1fr 1fr 1.3fr; gap:.8rem; margin:1rem 0 0; }
.tile { background:var(--soft); border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; }
.tile b { display:block; font-family:var(--display); font-size:1.5rem; line-height:1.15; color:var(--ink); }
.tile b.sm { font-size:1.08rem; line-height:1.3; padding-top:.2rem; }
.tile span { color:var(--muted); font-size:.93rem; }

.section-title { font-family:var(--display); font-size:1.25rem; font-weight:700; margin:1.9rem 0 .7rem; color:var(--ink); }
.bubble { background:var(--soft); border:1px solid var(--line); border-radius:18px 18px 18px 4px; padding:1rem 1.2rem; font-size:1.1rem; line-height:1.8; color:var(--ink); word-break:break-word; }
.bubble mark { background:#FFE08A; color:var(--ink); padding:.08em .2em; border-radius:4px; box-decoration-break:clone; -webkit-box-decoration-break:clone; }
.bubble sup { display:inline-block; background:var(--ink); color:#fff; font-size:.68rem; font-weight:700; line-height:1; min-width:1.05rem; padding:.17rem .26rem; margin-left:.15rem; border-radius:999px; text-align:center; vertical-align:.5em; }
.bubble-note { color:var(--muted); font-size:.93rem; margin:.5rem 0 0; }

.sign { display:grid; grid-template-columns:1.8rem 1fr auto; gap:.9rem; padding:.9rem 0; border-top:1px solid var(--line); align-items:start; }
.sign:first-of-type { border-top:none; padding-top:.2rem; }
.badge { width:1.7rem; height:1.7rem; border-radius:50%; background:var(--ink); color:#fff; font-weight:700; font-size:.9rem; text-align:center; line-height:1.7rem; }
.sign b { display:block; font-size:1.07rem; color:var(--ink); }
.sign .why { color:var(--muted); font-size:1rem; line-height:1.45; }
.strength { display:flex; align-items:center; gap:.5rem; color:var(--muted); font-size:.92rem; white-space:nowrap; padding-top:.15rem; }
.meter { display:flex; gap:3px; }
.meter i { display:block; width:16px; height:6px; border-radius:3px; background:var(--line); }
.m1 i:nth-child(-n+1), .m2 i:nth-child(-n+2), .m3 i:nth-child(-n+3) { background:var(--scam); }
.none { color:var(--muted); font-size:1.05rem; margin:.2rem 0 0; }

.todo { margin:0; padding:0; list-style:none; counter-reset:step; }
.todo li { counter-increment:step; position:relative; padding:.55rem 0 .55rem 2.6rem; font-size:1.07rem; line-height:1.5; color:var(--body); }
.todo li::before { content:counter(step); position:absolute; left:0; top:.55rem; width:1.7rem; height:1.7rem; border-radius:8px; background:#E4EBF4; color:var(--primary); font-weight:700; text-align:center; line-height:1.7rem; font-size:.95rem; }
.todo li b { color:var(--ink); }
.helpline { border-radius:12px; background:var(--primary); color:#fff; padding:1rem 1.2rem; margin-top:1rem; font-size:1.05rem; line-height:1.5; }
.helpline b { color:#fff; font-size:1.15rem; }
.helpline small { display:block; color:#C5D3E4; margin-top:.35rem; font-size:.9rem; }

/* empty state */
.empty h3 { font-family:var(--display); margin:0 0 .4rem; font-size:1.5rem; color:var(--ink); }
.empty > p { margin:0 0 1.1rem; color:var(--muted); font-size:1.08rem; line-height:1.55; }
.checklist { display:grid; grid-template-columns:1fr 1fr; gap:.55rem .9rem; }
.checklist div { display:flex; gap:.6rem; align-items:flex-start; font-size:1rem; line-height:1.4; color:var(--body); padding:.65rem .8rem; background:var(--soft); border:1px solid var(--line); border-radius:10px; }
.checklist div::before { content:""; flex:0 0 .55rem; height:.55rem; margin-top:.4rem; border-radius:50%; background:var(--primary); }

/* history */
.recent { display:flex; gap:.8rem; align-items:center; padding:.55rem 0; border-top:1px solid var(--line); font-size:1rem; }
.recent .tag { font-weight:700; font-size:.85rem; padding:.15rem .65rem; border-radius:999px; white-space:nowrap; }
.tag.scam { background:var(--scam-bg); color:var(--scam); }
.tag.suspicious { background:var(--warn-bg); color:var(--warn); }
.tag.safe { background:var(--safe-bg); color:var(--safe); }
.recent .snip { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* info pages */
.prose { max-width:46rem; font-size:1.08rem; line-height:1.65; }
.prose h3 { font-family:var(--display); font-size:1.4rem; margin:1.8rem 0 .5rem; color:var(--ink); }
.prose h3:first-child { margin-top:0; }
.prose li { margin:.35rem 0; color:var(--body); }
.flow { list-style:none; counter-reset:flow; padding:0; margin:1rem 0; max-width:46rem; }
.flow li { counter-increment:flow; position:relative; padding:1rem 0 1rem 3.4rem; border-top:1px solid var(--line); font-size:1.07rem; line-height:1.55; color:var(--body); }
.flow li::before { content:counter(flow); position:absolute; left:0; top:.8rem; font-family:var(--display); font-size:2rem; font-weight:700; line-height:1; color:var(--primary); }
.flow b { display:block; font-size:1.15rem; color:var(--ink); }
.metric-block { max-width:46rem; background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1rem 1.3rem; margin:.8rem 0; }
.metric-block h4 { margin:0 0 .6rem; font-family:var(--display); font-size:1.1rem; color:var(--ink); }
.bar-row { display:grid; grid-template-columns:5.5rem 1fr 3rem; gap:.8rem; align-items:center; padding:.25rem 0; font-size:1rem; color:var(--body); }
.bar { height:10px; border-radius:5px; background:var(--line); overflow:hidden; }
.bar i { display:block; height:100%; background:var(--primary); border-radius:5px; }
.delta { max-width:46rem; font-size:1.05rem; color:var(--body); }
.scam-grid { display:grid; grid-template-columns:1fr 1fr; gap:.9rem; max-width:60rem; }
.scam-card { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1rem 1.2rem; }
.scam-card h4 { margin:0 0 .35rem; font-family:var(--display); font-size:1.12rem; color:var(--ink); }
.scam-card p { margin:.2rem 0; font-size:1rem; line-height:1.5; color:var(--body); }
.scam-card p b { color:var(--ink); }

.footer { margin-top:2.5rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:.93rem; }

[data-testid="stExpander"] { background:var(--soft); border:1px solid var(--line); border-radius:12px; margin-top:1.4rem; }
[data-testid="stExpander"] summary p { font-weight:700; color:var(--ink) !important; }
[data-testid="stExpander"] summary { color:var(--ink); }
[data-testid="stAlert"] p { color:inherit; }

@media (max-width:768px) {
  .headline { font-size:1.9rem; } .sdg { display:none; } .verdict h2 { font-size:1.6rem; }
  .tiles, .checklist, .scam-grid { grid-template-columns:1fr; }
  .sign { grid-template-columns:1.8rem 1fr; } .sign .strength { grid-column:2; }
}
@media (prefers-reduced-motion:reduce) { * { transition:none !important; animation:none !important; } .dial::before { --a:var(--t); } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- knowledge
# weight: 1 = mild, 2 = moderate, 3 = strong
SIGNS = [
    dict(name="Asks you to click a link", weight=1,
         why="Scam texts use links to steal logins, card details or money.",
         pattern=r"https?://\S+|www\.\S+|\b[\w-]+\.(?:com|in|co|org|net|xyz|top|info)\b\S*"),
    dict(name="Uses a shortened or disguised link", weight=2,
         why="Short links hide where you will really land. Banks and couriers rarely use them.",
         pattern=r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|cutt\.ly|rb\.gy|is\.gd)\S*"),
    dict(name="Creates panic or a tight deadline", weight=1,
         why="Pressure stops you from thinking. Real organisations give you time.",
         pattern=r"\b(?:urgent(?:ly)?|immediately|right now|asap|today|tonight|last chance|expires?|within \d+ (?:hours?|minutes?|mins?))\b"),
    dict(name="Threatens to block or cut off a service", weight=2,
         why="Fake alerts claim your account, SIM, gas or electricity will stop unless you act.",
         pattern=r"\b(?:block(?:ed)?|suspend(?:ed)?|disconnect(?:ed)?|deactivat\w*|freez\w*|terminat\w*)\b"),
    dict(name="Asks you to pay a fee or send money", weight=3,
         why="Genuine jobs, prizes, refunds and deliveries do not ask for money first.",
         pattern=r"\b(?:pay|send|deposit|transfer|remit)\b[^.\n]{0,40}(?:rs\.?|\u20b9|inr|fee|charges?|registration|deposit)"
                 r"|\b(?:fee|charges?|registration)\b[^.\n]{0,25}\b(?:pay|upi|paytm|gpay|phonepe)\b"),
    dict(name="Asks for an OTP, PIN, password or Aadhaar", weight=3,
         why="No bank or company will ask you to share these. Anyone who does is a scammer.",
         pattern=r"\b(?:share|send|give|enter|tell|submit|reply|provide)\b[^.\n]{0,30}\b(?:otp|pin|password|cvv|aadhaar|card number)\b",
         unless=r"(?:do not|don't|dont|never)\s+share"),
    dict(name="Promises a prize, refund or easy money", weight=2,
         why="Unexpected winnings and refunds are the bait in most scams.",
         pattern=r"\b(?:lottery|you(?:'ve| have)? won|prize|winner|jackpot|cashback|refund|free gift|earn [^.\n]{0,15}(?:daily|per day|weekly))\b"),
    dict(name="Tells you to call an unknown number", weight=1,
         why="Scammers move you to a phone call, where they can pressure you live.",
         pattern=r"\bcall\b[^.\n]{0,30}\+?\d[\d\s-]{8,}\d"),
]

# name: (keywords matched as whole words, what to do, how the trick works)
TYPES = {
    "Fake bank or KYC alert": (["kyc", "bank", "account", "pan card", "blocked"],
        "Banks do not update KYC through SMS links. Visit the branch or use the bank's official app.",
        "Claims your account will be blocked unless you update KYC through a link, then copies your login."),
    "UPI or refund trap": (["upi", "refund", "pin", "cashback"],
        "You never need to enter a UPI PIN to receive money. A PIN request means money is going out.",
        "Promises a refund, then asks you to approve a request or enter a PIN, which sends money away."),
    "Lottery or prize claim": (["lottery", "prize", "won", "winner", "jackpot"],
        "Real prizes never ask you to pay a fee first. You cannot win a contest you did not enter.",
        "Says you won something big, then asks for a small fee or your Aadhaar to release it."),
    "Fake job offer": (["job", "earn", "registration", "salary", "work from home"],
        "Genuine employers do not charge registration or training fees.",
        "Offers easy work-from-home pay, then charges a registration or training fee."),
    "Courier or customs fee": (["parcel", "courier", "customs", "delivery", "package"],
        "Check the courier's official app or website with your tracking number, not the SMS link.",
        "Says a parcel is stuck and asks for a small fee on a fake payment page."),
    "Electricity or utility threat": (["electricity", "disconnected", "power", "bill", "gas"],
        "Check your bill on the official board website or app. Utilities do not cut power over an SMS.",
        "Threatens tonight's power cut and pushes you to call a number or install an app."),
}

EXAMPLES = {
    "Bank account block": "Dear customer your bank account will be blocked today. Update KYC now at http://bit.ly/kyc-sbi-update",
    "Electricity cut": "Your electricity will be disconnected tonight 9:30pm because previous bill is unpaid. Call 9876543210 immediately.",
    "Prize message": "Congratulations! You won Rs 25,00,000 in the lucky draw. Send Aadhaar and Rs 5000 fee to claim your prize.",
    "Genuine OTP": "Your OTP for login is 482913. Do not share it with anyone. Valid for 5 minutes.",
    "Message from a friend": "Hi, are we still meeting at 6pm near the college gate?",
}

LEVELS = {
    "scam": dict(css="v-scam", title="This looks like a scam.",
                 body="Do not click, reply or pay anything. Check the steps below."),
    "suspicious": dict(css="v-suspicious", title="This message has warning signs.",
                       body="Do not act on it yet. Confirm through an official channel first."),
    "safe": dict(css="v-safe", title="No common scam tricks found.",
                 body="This looks routine. Even so, never share an OTP, PIN or password with anyone."),
}
STRENGTH = {1: "Mild", 2: "Moderate", 3: "Strong"}
LEVEL_NAME = {"scam": "High risk", "suspicious": "Caution", "safe": "Low risk"}


# ---------------------------------------------------------------- model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


def clean(text: str) -> str:
    """Must match the cleaning used in the training notebook exactly."""
    t = text.lower()
    t = re.sub(r"(https?://\S+|www\.\S+|\b\w+\.(com|in|ly|co|org)\S*)", " urltoken ", t)
    t = re.sub(r"\+?\d[\d\s-]{8,}\d", " phonetoken ", t)
    t = re.sub(r"\d+", " numtoken ", t)
    return t


def model_leaning(model, text: str, k: int = 5):
    """Words that pushed the language model toward 'scam' for this message."""
    try:
        vec, clf = model.steps[0][1], model.steps[-1][1]
        row = vec.transform([clean(text)]).multiply(clf.coef_[0]).tocoo()
        names = vec.get_feature_names_out()
        ranked = sorted(zip(row.data, row.col), reverse=True)
        return [names[c] for v, c in ranked if v > 0 and "token" not in names[c]][:k]
    except Exception:
        return []  # never let this optional detail break the page


def analyse(model, text: str) -> dict:
    text = text[:1000]
    found = []
    for sign in SIGNS:
        matches = list(re.finditer(sign["pattern"], text, re.I))
        if sign.get("unless") and re.search(sign["unless"], text, re.I):
            matches = []
        if matches:
            found.append(dict(sign, spans=[m.span() for m in matches]))
    found.sort(key=lambda s: -s["weight"])           # strongest first; the number shown = position in this list
    spans = [(a, b, i + 1) for i, s in enumerate(found) for a, b in s["spans"]]

    p = float(model.predict_proba([clean(text)])[0][1])
    rule = min(1.0, sum(s["weight"] for s in found) / 7)
    score = 0.65 * rule + 0.35 * p
    if p >= 0.90:                       # a very confident model alone earns at least a caution
        score = max(score, 0.40)
    level = "scam" if score >= 0.60 else "suspicious" if score >= 0.38 else "safe"

    low = text.lower()
    best, best_hits = None, 0
    for name, (words, tip, _) in TYPES.items():
        hits = sum(bool(re.search(r"\b" + re.escape(w) + r"\b", low)) for w in words)  # whole words only
        if hits > best_hits:
            best, best_hits = (name, tip), hits

    return dict(text=text, level=level, score=round(score * 100), prob=p, signs=found,
                spans=spans, kind=best if level != "safe" else None,
                words=model_leaning(model, text))


# ---------------------------------------------------------------- rendering
def h(markup: str) -> str:
    """Collapse indentation so Streamlit's markdown never reads our HTML as a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def esc(s: str) -> str:
    """Escape user text for HTML. '$' is escaped too so Streamlit does not treat it as maths."""
    return html.escape(s).replace("$", "&#36;")


def highlight(text: str, spans) -> str:
    """Mark concerning phrases and tag each with the number of the sign that flagged it."""
    merged = []
    for s, e, n in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2].add(n)
        else:
            merged.append([s, e, {n}])
    out, pos = [], 0
    for s, e, nums in merged:
        out.append(esc(text[pos:s]))
        tags = "".join(f"<sup>{n}</sup>" for n in sorted(nums))
        out.append(f"<mark>{esc(text[s:e])}</mark>{tags}")
        pos = e
    out.append(esc(text[pos:]))
    return "".join(out).replace("\r", "").replace("\n", "<br>")


def report_text(r: dict) -> str:
    lines = ["SCAM SMS SHIELD - CHECK REPORT", "", "Message:", r["text"], "",
             f"Verdict: {LEVELS[r['level']]['title']} ({LEVEL_NAME[r['level']]})",
             f"Risk score: {r['score']} / 100",
             f"Wording similarity to known scams: {round(r['prob'] * 100)}%", ""]
    if r["kind"]:
        lines += [f"Looks like: {r['kind'][0]}", ""]
    lines.append("Warning signs found:")
    lines += [f"  {i}. {s['name']} ({STRENGTH[s['weight']]}) - {s['why']}" for i, s in enumerate(r["signs"], 1)] or ["  None"]
    lines += ["", "If you already paid or shared details: call 1930 and report at cybercrime.gov.in",
              "This is a student prototype. Confirm with your bank or the police."]
    return "\n".join(lines)


def render_result(r: dict):
    lv = LEVELS[r["level"]]
    kind = r["kind"][0] if r["kind"] else "No known scam type"
    st.markdown(h(f"""
        <div class="verdict {lv['css']}">
          <div class="vtext"><h2>{lv['title']}</h2><p>{lv['body']}</p></div>
          <div class="dial" style="--t:{r['score'] * 1.8}deg"><div class="num"><b>{r['score']}</b><span>out of 100</span></div></div>
        </div>
        <div class="tiles">
          <div class="tile"><b>{len(r['signs'])} of {len(SIGNS)}</b><span>Warning signs found</span></div>
          <div class="tile"><b>{round(r['prob'] * 100)}%</b><span>Wording matches scams</span></div>
          <div class="tile"><b class="sm">{kind}</b><span>Most likely type</span></div>
        </div>"""), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Your message, with the concerning parts marked</div>', unsafe_allow_html=True)
    note = "Each number matches a warning sign in the list below." if r["spans"] else "Nothing in this message matched a known scam trick."
    st.markdown(h(f"""<div class="bubble">{highlight(r['text'], r['spans'])}</div><p class="bubble-note">{note}</p>"""),
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Why we reached this verdict</div>', unsafe_allow_html=True)
    if r["signs"]:
        rows = "".join(
            f'<div class="sign"><div class="badge">{i}</div>'
            f'<div><b>{s["name"]}</b><span class="why">{s["why"]}</span></div>'
            f'<div class="strength"><span class="meter m{s["weight"]}"><i></i><i></i><i></i></span>{STRENGTH[s["weight"]]}</div></div>'
            for i, s in enumerate(r["signs"], 1))
        st.markdown(h(rows), unsafe_allow_html=True)
    else:
        st.markdown('<p class="none">No warning signs were found in this message.</p>', unsafe_allow_html=True)

    if r["level"] != "safe":
        st.markdown('<div class="section-title">What to do now</div>', unsafe_allow_html=True)
        lead = f"<li><b>{r['kind'][0]}.</b> {r['kind'][1]}</li>" if r["kind"] else ""
        st.markdown(h(f"""<ol class="todo">{lead}
            <li>Do not click the link or call the number in the message.</li>
            <li>Never share an OTP, PIN, password or card details with anyone.</li>
            <li>Check using the official app, or the number printed on your card or bill.</li>
            <li>Tell a family member, especially if the message came with pressure to act fast.</li></ol>
            <div class="helpline">Already paid or shared details? Call <b>1930</b> (National Cyber Crime Helpline) right away
            and report at <b>cybercrime.gov.in</b>.<small>Details are from the Government of India. Confirm them on the official portal.</small></div>"""),
            unsafe_allow_html=True)

    st.download_button("Save this report", report_text(r), file_name="scam-check-report.txt", mime="text/plain")

    with st.expander("Model details for reviewers"):
        st.write(f"Language model: TF-IDF (single words and two-word phrases) with logistic regression. "
                 f"Scam probability for this text: **{r['prob']:.1%}**.")
        st.write("Final risk score = 65% from the weighted warning signs + 35% from the language model. "
                 "Under 38 is low risk, 38 to 59 is caution, 60 and above is high risk.")
        if r["words"]:
            st.write("Words the model leaned on most: " + ", ".join(r["words"]))


def render_empty():
    items = "".join(f"<div>{s['name']}</div>" for s in SIGNS)
    st.markdown(h(f"""<div class="empty"><h3>Your result will appear here</h3>
        <p>Paste a message and choose Check message. You will see which parts raised concern, why, and what to do next.</p>
        <div class="checklist">{items}</div></div>"""), unsafe_allow_html=True)


# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("msg", "")
ss.setdefault("result", None)
ss.setdefault("history", [])
ss.setdefault("pick", None)
ss.setdefault("run", False)
ss.setdefault("empty_note", False)


def load_example():
    """Chosen example fills the box and is checked straight away."""
    if ss.pick in EXAMPLES:
        ss.msg, ss.run, ss.empty_note = EXAMPLES[ss.pick], True, False
    ss.pick = None


def clear_all():
    ss.msg, ss.result, ss.pick, ss.empty_note, ss.run = "", None, None, False, False


# ---------------------------------------------------------------- page
st.markdown(h("""<div class="masthead"><div class="brand"><span class="crest"></span><span class="wordmark">Scam SMS Shield</span></div>
    <span class="sdg">SDG 16: Peace, Justice and Strong Institutions</span></div>
    <h1 class="headline">Is this message a scam?</h1>
    <p class="lede">Paste the text you received. We look for the tricks scammers rely on, show you the proof, and tell you what to do next.</p>"""),
    unsafe_allow_html=True)

try:
    model = load_model()
except FileNotFoundError:
    st.error("The model file model.joblib was not found. Add it to the same folder as app.py and reload.")
    st.stop()
except Exception as e:  # e.g. the model was saved with a different scikit-learn version
    st.error("The model file could not be loaded. Re-save it in Colab and upload it again.")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()

tab_check, tab_how, tab_types, tab_about = st.tabs(["Check a message", "How it works", "Common scams", "About and limits"])

with tab_check:
    left, right = st.columns([5, 6], gap="large")

    with left:
        with st.container(key="input"):
            st.markdown('<p class="panel-title">Message you received</p>', unsafe_allow_html=True)
            st.text_area("Message", key="msg", height=200, max_chars=1000, label_visibility="collapsed",
                         placeholder="Paste the full text of the SMS here")
            st.markdown('<p class="small-label">No message handy? Try an example.</p>', unsafe_allow_html=True)
            with st.container(key="picker"):
                st.pills("Examples", list(EXAMPLES), key="pick", on_change=load_example, label_visibility="collapsed")
            b1, b2 = st.columns([3, 2])
            go = b1.button("Check message", type="primary", use_container_width=True)
            b2.button("Clear", on_click=clear_all, use_container_width=True)

            if go or ss.run:
                ss.run = False
                text = ss.msg.strip()
                if text:
                    ss.empty_note = False
                    ss.result = analyse(model, text)
                    ss.history = (ss.history + [dict(text=text, level=ss.result["level"])])[-5:]
                else:
                    ss.empty_note, ss.result = True, None
            if ss.empty_note:
                st.markdown('<p class="notice">Paste a message first, then choose Check message.</p>', unsafe_allow_html=True)

            if ss.history:
                st.markdown('<div class="section-title">Checked in this session</div>', unsafe_allow_html=True)
                for item in reversed(ss.history):
                    snip = esc(item["text"][:70] + ("..." if len(item["text"]) > 70 else ""))
                    st.markdown(f'<div class="recent"><span class="tag {item["level"]}">{LEVEL_NAME[item["level"]]}</span>'
                                f'<span class="snip">{snip}</span></div>', unsafe_allow_html=True)

    with right:
        with st.container(key="result"):
            if ss.result:
                render_result(ss.result)
            else:
                render_empty()

with tab_how:
    st.markdown(h("""<div class="prose"><h3>Two checks, one clear answer</h3>
        <p>A language model on its own can be right for the wrong reasons. So every message is checked two ways, and you can see both.</p></div>
        <ol class="flow">
        <li><b>Clean the text</b>Links, phone numbers and amounts are swapped for neutral placeholders. The model then learns from wording, not from digits.</li>
        <li><b>Read the wording</b>TF-IDF counts single words and two-word phrases and gives extra weight to the ones that are rare but telling.</li>
        <li><b>Score it two ways</b>Logistic regression rates how much the wording resembles known scams. Eight rule-based warning signs check what the message asks you to do.</li>
        <li><b>Show the evidence</b>The concerning phrases are highlighted and numbered in your own message, each sign is explained, and the next steps depend on the type of scam.</li></ol>"""),
        unsafe_allow_html=True)

    rows = [(n, p, r, f) for n, p, r, f in METRICS if p is not None]
    if rows:
        blocks = ""
        for n, p, r, f in rows:
            bars = "".join(f'<div class="bar-row"><span>{lab}</span><div class="bar"><i style="width:{v * 100:.0f}%"></i></div><b>{v:.2f}</b></div>'
                           for lab, v in (("Precision", p), ("Recall", r), ("F1", f)))
            blocks += f'<div class="metric-block"><h4>{n}</h4>{bars}</div>'
        delta = ""
        if len(rows) >= 2:
            delta = f'<p class="delta">Adding Indian examples changed F1 from <b>{rows[0][3]:.2f}</b> to <b>{rows[-1][3]:.2f}</b>.</p>'
        st.markdown(h(f"""<div class="prose"><h3>Results on Indian-style scam messages</h3>
            <p>Measured on messages the model never saw during training.</p></div>{blocks}{delta}
            <div class="prose"><p>Precision answers: of the messages we flagged, how many were truly scams. Recall answers: of all real scams, how many we caught.</p></div>"""),
            unsafe_allow_html=True)

with tab_types:
    cards = "".join(f'<div class="scam-card"><h4>{n}</h4><p><b>How it works.</b> {how}</p><p><b>What to do.</b> {tip}</p></div>'
                    for n, (_, tip, how) in TYPES.items())
    st.markdown(h(f"""<div class="prose"><h3>Six scams to know</h3>
        <p>These are the patterns the checker is built around. If a message fits one of them, treat it with care.</p></div>
        <div class="scam-grid">{cards}</div>"""), unsafe_allow_html=True)

with tab_about:
    st.markdown(h("""<div class="prose"><h3>Why this exists</h3>
        <p>Fraud by SMS hits people who are new to smartphones the hardest, often the elderly and first-time users.
        This project supports Sustainable Development Goal 16, Peace, Justice and Strong Institutions, by helping people spot fraud before it costs them.</p>
        <h3>What it cannot do</h3><ul>
        <li>It was trained mostly on older English messages plus a small set of Indian-style examples, so new scam wording can slip through.</li>
        <li>It can raise a caution on a genuine message that contains a link or a deadline. When unsure, confirm through the official app.</li>
        <li>It reads text only. It cannot check who sent the message.</li>
        <li>It is a student prototype and not a substitute for advice from your bank or the police.</li></ul>
        <h3>Your privacy</h3>
        <p>This app does not save the messages you paste. They are used only to produce the result on your screen.</p></div>"""),
        unsafe_allow_html=True)

st.markdown(f'<div class="footer">Academic capstone project by {esc(AUTHOR)}. Machine Learning and Applied AI internship, AICT x IBM SkillsBuild.</div>',
            unsafe_allow_html=True)
