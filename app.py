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

st.set_page_config(page_title="Scam SMS Shield", layout="wide", initial_sidebar_state="collapsed")

# Fill these in with your own results from Colab (precision, recall, F1 as 0-1 values).
# Rows left as None are hidden from the page.
METRICS = [
    ("Public data only", None, None, None),
    ("With Indian examples added", None, None, None),
]

# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');

:root {
  --bg:#EDF1F5; --surface:#FFFFFF; --ink:#0F1B2D; --muted:#566377; --line:#D3DAE3; --primary:#1B3A5C;
  --scam:#B42318; --scam-bg:#FDECEA; --scam-line:#F3B8B1;
  --warn:#8A5A00; --warn-bg:#FFF4D6; --warn-line:#EBCB7C;
  --safe:#14633F; --safe-bg:#E5F4EC; --safe-line:#A9D8BE;
}
html, body, .stApp, .stApp * { font-family:'Atkinson Hyperlegible', system-ui, sans-serif; }
.stApp { background:var(--bg); color:var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
.block-container { max-width:1120px; padding:1.6rem 1.5rem 3rem; }

.masthead { display:flex; justify-content:space-between; align-items:baseline; border-bottom:2px solid var(--ink); padding-bottom:.6rem; }
.wordmark { font-weight:700; font-size:1.15rem; letter-spacing:.01em; }
.tagline { color:var(--muted); font-size:.95rem; }
.headline { font-size:3rem; line-height:1.08; font-weight:700; margin:2.2rem 0 .6rem; letter-spacing:-.01em; }
.lede { font-size:1.2rem; line-height:1.55; color:var(--muted); max-width:44rem; margin:0 0 1.6rem; }

.stTabs [data-baseweb="tab-list"] { gap:1.8rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding:.7rem 0; background:transparent; font-size:1.05rem; color:var(--muted); }
.stTabs [aria-selected="true"] { color:var(--ink); font-weight:700; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--ink); height:3px; }

.field-label { font-weight:700; font-size:1.02rem; margin:0 0 .4rem; }
[data-testid="stTextArea"] textarea {
  background:var(--surface); border:2px solid var(--line); border-radius:6px; font-size:1.08rem; line-height:1.55; color:var(--ink); padding:.9rem 1rem;
}
[data-testid="stTextArea"] textarea:focus { border-color:var(--primary); box-shadow:0 0 0 3px rgba(27,58,92,.18); }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background:var(--surface); border:2px solid var(--line); border-radius:6px; min-height:2.8rem; }
[data-testid="stSelectbox"] label p, [data-testid="stTextArea"] label p { font-weight:700; }

.stButton > button { border-radius:6px; font-weight:700; font-size:1.05rem; min-height:3rem; padding:.55rem 1.3rem; transition:background .15s, color .15s; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] { background:var(--primary); color:#fff; border:2px solid var(--primary); }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover { background:var(--ink); border-color:var(--ink); color:#fff; }
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] { background:transparent; color:var(--ink); border:2px solid var(--line); }
.stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="stBaseButton-secondary"]:hover { border-color:var(--ink); color:var(--ink); }
.stButton > button:focus-visible { outline:3px solid var(--primary); outline-offset:2px; }

.notice { color:var(--scam); font-weight:700; margin:.4rem 0 0; }

.verdict { padding:1.3rem 1.5rem; border-left:8px solid; border-radius:2px 10px 10px 2px; }
.verdict h2 { font-size:2rem; line-height:1.15; margin:0 0 .35rem; font-weight:700; }
.verdict p { margin:0; font-size:1.12rem; line-height:1.5; }
.v-scam { background:var(--scam-bg); border-color:var(--scam); color:#5E130C; }
.v-suspicious { background:var(--warn-bg); border-color:var(--warn); color:#4A3000; }
.v-safe { background:var(--safe-bg); border-color:var(--safe); color:#0B3A25; }

.gauge { position:relative; margin:1.4rem 0 .3rem; }
.zones { display:flex; height:12px; border-radius:6px; overflow:hidden; }
.zones span:nth-child(1) { width:38%; background:#9ED3B5; }
.zones span:nth-child(2) { width:22%; background:#EBCB7C; }
.zones span:nth-child(3) { width:40%; background:#E8948B; }
.pin { position:absolute; top:-7px; width:4px; height:26px; background:var(--ink); border-radius:2px; transform:translateX(-50%); }
.scale { display:flex; justify-content:space-between; color:var(--muted); font-size:.9rem; }

.facts { display:flex; gap:2rem; margin:1.2rem 0 0; padding:.9rem 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); flex-wrap:wrap; }
.facts div { min-width:7rem; }
.facts b { display:block; font-size:1.6rem; line-height:1.1; }
.facts span { color:var(--muted); font-size:.95rem; }

.section-title { font-size:1.3rem; font-weight:700; margin:1.8rem 0 .7rem; }
.bubble { background:var(--surface); border:2px solid var(--line); border-radius:20px 20px 20px 4px; padding:1rem 1.2rem; font-size:1.12rem; line-height:1.65; max-width:36rem; word-break:break-word; }
.bubble mark { background:#FFD966; color:var(--ink); padding:.05em .2em; border-radius:3px; }
.bubble-note { color:var(--muted); font-size:.95rem; margin:.5rem 0 0; }

.sign { display:grid; grid-template-columns:14px 1fr auto; gap:.9rem; padding:.85rem 0; border-top:1px solid var(--line); align-items:start; }
.sign:first-of-type { border-top:none; }
.meter { display:flex; flex-direction:column-reverse; gap:3px; margin-top:.3rem; }
.meter i { display:block; width:14px; height:6px; border-radius:1px; background:var(--line); }
.meter.m1 i:nth-child(-n+1), .meter.m2 i:nth-child(-n+2), .meter.m3 i:nth-child(-n+3) { background:var(--scam); }
.sign b { display:block; font-size:1.08rem; }
.sign span.why { color:var(--muted); font-size:1rem; line-height:1.45; }
.sign .strength { color:var(--muted); font-size:.95rem; white-space:nowrap; }
.none { color:var(--muted); font-size:1.05rem; margin:.2rem 0 0; }

.todo { margin:0; padding:0; list-style:none; counter-reset:step; }
.todo li { counter-increment:step; position:relative; padding:.55rem 0 .55rem 2.6rem; font-size:1.08rem; line-height:1.5; }
.todo li::before { content:counter(step); position:absolute; left:0; top:.5rem; width:1.7rem; height:1.7rem; border-radius:50%; background:var(--ink); color:#fff; font-weight:700; text-align:center; line-height:1.7rem; font-size:.95rem; }
.helpline { border-left:6px solid var(--primary); background:var(--surface); padding:.9rem 1.1rem; margin-top:1rem; font-size:1.05rem; line-height:1.5; }
.helpline small { display:block; color:var(--muted); margin-top:.3rem; font-size:.9rem; }

.empty { border:2px dashed var(--line); border-radius:10px; padding:1.6rem 1.5rem; }
.empty h3 { margin:0 0 .5rem; font-size:1.4rem; }
.empty p { margin:.2rem 0; color:var(--muted); font-size:1.08rem; line-height:1.55; }

.recent { display:flex; gap:.8rem; align-items:baseline; padding:.5rem 0; border-top:1px solid var(--line); font-size:1rem; }
.recent .tag { font-weight:700; font-size:.9rem; padding:.1rem .6rem; border-radius:20px; white-space:nowrap; }
.tag.scam { background:var(--scam-bg); color:var(--scam); }
.tag.suspicious { background:var(--warn-bg); color:var(--warn); }
.tag.safe { background:var(--safe-bg); color:var(--safe); }
.recent span.snip { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

.prose { max-width:46rem; font-size:1.1rem; line-height:1.65; }
.prose h3 { font-size:1.4rem; margin:1.8rem 0 .5rem; }
.prose li { margin:.35rem 0; }
.flow { list-style:none; counter-reset:flow; padding:0; margin:1rem 0; max-width:46rem; }
.flow li { counter-increment:flow; position:relative; padding:.9rem 0 .9rem 3.2rem; border-top:1px solid var(--line); font-size:1.08rem; line-height:1.55; }
.flow li::before { content:counter(flow); position:absolute; left:0; top:.8rem; font-size:2rem; font-weight:700; line-height:1; color:var(--primary); }
.flow b { display:block; font-size:1.15rem; }
table.results { border-collapse:collapse; width:100%; max-width:46rem; font-size:1.05rem; margin-top:.5rem; }
table.results th, table.results td { text-align:left; padding:.65rem .8rem .65rem 0; border-bottom:1px solid var(--line); }
table.results th { color:var(--muted); font-weight:400; }

.footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:.95rem; }

[data-testid="stExpander"] { background:var(--surface); border:1px solid var(--line); border-radius:6px; margin-top:1.4rem; }
[data-testid="stExpander"] summary { font-weight:700; }

@media (max-width:768px) { .headline { font-size:2.2rem; } .tagline { display:none; } .verdict h2 { font-size:1.6rem; } }
@media (prefers-reduced-motion:reduce) { * { transition:none !important; } }
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

TYPES = {
    "Fake bank or KYC alert": (["kyc", "bank", "account", "pan card", "blocked"],
        "Banks do not update KYC through SMS links. Visit the branch or use the bank's official app."),
    "UPI or refund trap": (["upi", "refund", "pin", "cashback"],
        "You never need to enter a UPI PIN to receive money. A PIN request means money is going out."),
    "Lottery or prize claim": (["lottery", "prize", "won", "winner", "jackpot"],
        "Real prizes never ask you to pay a fee first. You cannot win a contest you did not enter."),
    "Fake job offer": (["job", "earn", "registration", "salary", "work from home"],
        "Genuine employers do not charge registration or training fees."),
    "Courier or customs fee": (["parcel", "courier", "customs", "delivery", "package"],
        "Check the courier's official app or website with your tracking number, not the SMS link."),
    "Electricity or utility threat": (["electricity", "disconnected", "power", "bill", "gas"],
        "Check your bill on the official board website or app. Utilities do not cut power over an SMS."),
}

EXAMPLES = {
    "Bank account block": "Dear customer your bank account will be blocked today. Update KYC now at http://bit.ly/kyc-sbi-update",
    "Electricity cut": "Your electricity will be disconnected tonight 9:30pm because previous bill is unpaid. Call 9876543210 immediately.",
    "Prize message": "Congratulations! You won Rs 25,00,000 in the lucky draw. Send Aadhaar and Rs 5000 fee to claim your prize.",
    "Genuine OTP": "Your OTP for login is 482913. Do not share it with anyone. Valid for 5 minutes.",
    "Friend": "Hi, are we still meeting at 6pm near the college gate?",
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
    vec, clf = model.steps[0][1], model.steps[1][1]
    row = vec.transform([clean(text)]).multiply(clf.coef_[0]).tocoo()
    names = vec.get_feature_names_out()
    ranked = sorted(zip(row.data, row.col), reverse=True)
    return [names[c] for v, c in ranked if v > 0 and "token" not in names[c]][:k]


def analyse(model, text: str) -> dict:
    found, spans = [], []
    for sign in SIGNS:
        matches = list(re.finditer(sign["pattern"], text, re.I))
        if sign.get("unless") and re.search(sign["unless"], text, re.I):
            matches = []
        if matches:
            found.append(sign)
            spans += [m.span() for m in matches]

    p = float(model.predict_proba([clean(text)])[0][1])
    rule = min(1.0, sum(s["weight"] for s in found) / 7)
    score = 0.65 * rule + 0.35 * p
    if p >= 0.90:                       # a very confident model alone earns at least a caution
        score = max(score, 0.40)
    level = "scam" if score >= 0.60 else "suspicious" if score >= 0.38 else "safe"

    low = text.lower()
    best, best_hits = None, 0
    for name, (words, tip) in TYPES.items():
        hits = sum(w in low for w in words)
        if hits > best_hits:
            best, best_hits = (name, tip), hits

    return dict(text=text, level=level, score=round(score * 100), prob=p, signs=found,
                spans=spans, kind=best if level != "safe" else None,
                words=model_leaning(model, text))


# ---------------------------------------------------------------- rendering
def h(markup: str) -> str:
    """Collapse indentation so Streamlit's markdown never reads our HTML as a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def highlight(text: str, spans) -> str:
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, pos = [], 0
    for s, e in merged:
        out.append(html.escape(text[pos:s]))
        out.append("<mark>" + html.escape(text[s:e]) + "</mark>")
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def render_result(r: dict):
    lv = LEVELS[r["level"]]
    st.markdown(h(f"""
        <div class="verdict {lv['css']}"><h2>{lv['title']}</h2><p>{lv['body']}</p></div>
        <div class="gauge"><div class="zones"><span></span><span></span><span></span></div>
        <div class="pin" style="left:{max(1, min(99, r['score']))}%"></div></div>
        <div class="scale"><span>Low risk</span><span>Caution</span><span>High risk</span></div>
        <div class="facts">
          <div><b>{r['score']} / 100</b><span>Risk score</span></div>
          <div><b>{len(r['signs'])} of {len(SIGNS)}</b><span>Warning signs found</span></div>
          <div><b>{round(r['prob'] * 100)}%</b><span>Wording looks like scams</span></div>
        </div>"""), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Your message, with the concerning parts marked</div>', unsafe_allow_html=True)
    st.markdown(h(f"""<div class="bubble">{highlight(r['text'], r['spans'])}</div>
        <p class="bubble-note">{'Highlighted phrases are what raised concern.' if r['spans'] else 'Nothing in this message matched a known scam trick.'}</p>"""),
        unsafe_allow_html=True)

    st.markdown('<div class="section-title">Why we reached this verdict</div>', unsafe_allow_html=True)
    if r["signs"]:
        rows = "".join(
            f'<div class="sign"><div class="meter m{s["weight"]}"><i></i><i></i><i></i></div>'
            f'<div><b>{s["name"]}</b><span class="why">{s["why"]}</span></div>'
            f'<span class="strength">{STRENGTH[s["weight"]]}</span></div>'
            for s in sorted(r["signs"], key=lambda s: -s["weight"]))
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
            and report at <b>cybercrime.gov.in</b>.<small>Contact details are from the Government of India. Confirm them on the official portal.</small></div>"""),
            unsafe_allow_html=True)

    with st.expander("Model details for reviewers"):
        st.write(f"Language model: TF-IDF (single words and two-word phrases) with logistic regression. "
                 f"Scam probability for this text: **{r['prob']:.1%}**.")
        st.write("Final risk score = 65% from the weighted warning signs + 35% from the language model. "
                 "Under 38 is low risk, 38 to 59 is caution, 60 and above is high risk.")
        if r["words"]:
            st.write("Words the model leaned on most: " + ", ".join(r["words"]))


def render_empty():
    st.markdown(h("""<div class="empty"><h3>Your result will appear here</h3>
        <p>Paste a message on the left and choose Check message.</p>
        <p>We look for eight tricks scammers rely on, such as fake deadlines, requests for an OTP, and disguised links.
        You will see exactly which parts of the message caused concern and what to do about it.</p></div>"""),
        unsafe_allow_html=True)


# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("msg", "")
ss.setdefault("result", None)
ss.setdefault("history", [])
ss.setdefault("example", "Choose an example")
ss.setdefault("empty_note", False)


def load_example():
    pick = ss.example
    if pick in EXAMPLES:
        ss.msg, ss.result, ss.empty_note = EXAMPLES[pick], None, False


def clear_all():
    ss.msg, ss.result, ss.example, ss.empty_note = "", None, "Choose an example", False


# ---------------------------------------------------------------- page
st.markdown(h("""<div class="masthead"><span class="wordmark">Scam SMS Shield</span>
    <span class="tagline">A free message check, built for SDG 16: safer communities</span></div>
    <h1 class="headline">Is this message a scam?</h1>
    <p class="lede">Paste the text you received. We look for the tricks scammers rely on and tell you what to do next.</p>"""),
    unsafe_allow_html=True)

try:
    model = load_model()
except FileNotFoundError:
    st.error("The model file model.joblib was not found. Add it to the same folder as app.py and reload.")
    st.stop()

tab_check, tab_how, tab_about = st.tabs(["Check a message", "How it works", "About and limits"])

with tab_check:
    left, right = st.columns([5, 6], gap="large")

    with left:
        st.markdown('<p class="field-label">Message you received</p>', unsafe_allow_html=True)
        st.text_area("Message", key="msg", height=210, label_visibility="collapsed",
                     placeholder="Paste the full text of the SMS here")
        st.selectbox("Or try an example", ["Choose an example"] + list(EXAMPLES), key="example", on_change=load_example)
        b1, b2 = st.columns([3, 2])
        go = b1.button("Check message", type="primary", use_container_width=True)
        b2.button("Clear", on_click=clear_all, use_container_width=True)
        if go:
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
                snip = html.escape(item["text"][:70] + ("..." if len(item["text"]) > 70 else ""))
                st.markdown(f'<div class="recent"><span class="tag {item["level"]}">{LEVEL_NAME[item["level"]]}</span>'
                            f'<span class="snip">{snip}</span></div>', unsafe_allow_html=True)

    with right:
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
        <li><b>Show the evidence</b>The concerning phrases are highlighted in your own message, each sign is explained, and the next steps depend on the type of scam.</li></ol>"""),
        unsafe_allow_html=True)

    rows = [(n, p, r, f) for n, p, r, f in METRICS if p is not None]
    if rows:
        body = "".join(f"<tr><td>{n}</td><td>{p:.2f}</td><td>{r:.2f}</td><td>{f:.2f}</td></tr>" for n, p, r, f in rows)
        st.markdown(h(f"""<div class="prose"><h3>Results on Indian-style scam messages</h3>
            <p>Measured on messages the model never saw during training.</p></div>
            <table class="results"><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{body}</table>
            <div class="prose"><p>Precision answers: of the messages we flagged, how many were truly scams. Recall answers: of all real scams, how many we caught.</p></div>"""),
            unsafe_allow_html=True)

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

st.markdown('<div class="footer">Academic capstone project, Machine Learning and Applied AI internship.</div>', unsafe_allow_html=True)
