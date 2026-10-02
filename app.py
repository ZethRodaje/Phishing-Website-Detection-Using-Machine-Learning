"""Phishing Website Detection - Streamlit app.

Loads phishing_model.joblib (saved by section 13 of the notebook) and scores URLs
using the same host-only features the model was trained on.

Run:  streamlit run app.py
"""
import math
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "phishing_model.joblib"
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
SUSP_WORDS = ["login", "signin", "secure", "verify", "account", "update", "banking", "confirm",
              "webscr", "password", "support", "billing", "recover", "unlock", "alert", "suspend"]
BORDERLINE_MARGIN = 0.15  # flag results this close to the decision threshold

EXAMPLES = {
    "Choose an example...": "",
    "google.com": "google.com",
    "github.com": "github.com",
    "http://192.168.10.5/login": "http://192.168.10.5/login",
    "http://secure-login-paypal.verify-account.xyz/update":
        "http://secure-login-paypal.verify-account.xyz/update",
}


# ---- feature code: identical to the notebook (sections 3.5 and 13) ----
def normalize_host(u):
    u = u.strip()
    if "://" not in u:
        u = "http://" + u
    h = urlparse(u).netloc.split(":")[0].lower()
    if h.startswith("www.") and h.count(".") >= 2:
        h = h[4:]
    return h


def shannon_entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def get_tld(host):
    if IP_RE.match(host):
        return "ip_address"
    parts = host.split(".")
    return parts[-1] if len(parts) >= 2 else "unknown"


def host_features(h):
    is_ip = bool(IP_RE.match(h))
    labels = h.split(".") if h else [""]
    n = max(len(h), 1)
    n_digits = len(re.findall(r"\d", h))
    n_hyphens = h.count("-")
    letters = re.findall(r"[a-z]", h)
    vowels = [ch for ch in letters if ch in "aeiou"]
    digit_runs = re.findall(r"\d+", h)
    return {
        "url_length": len(h),
        "num_dots": h.count("."),
        "has_ip": int(is_ip),
        "digits_count": n_digits,
        "num_hyphens": n_hyphens,
        "subdomain_count": 0 if is_ip else max(len(labels) - 2, 0),
        "entropy": shannon_entropy(h),
        "num_labels": len(labels),
        "longest_label": max(len(x) for x in labels),
        "longest_digit_run": max((len(x) for x in digit_runs), default=0),
        "digit_ratio": n_digits / n,
        "hyphen_ratio": n_hyphens / n,
        "vowel_ratio": len(vowels) / len(letters) if letters else 0.0,
        "has_susp_word": int(any(w in h for w in SUSP_WORDS)),
        "tld": get_tld(h),
    }


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def score(url, art):
    host = normalize_host(url)
    f = host_features(host)
    row = {c: f[c] for c in art["feature_order"] if c != "tld_freq"}
    row["tld_freq"] = art["tld_freq_map"].get(f["tld"], 0.0)
    X = pd.DataFrame([row])[art["feature_order"]]
    p = float(art["model"].predict_proba(X)[0, 1])
    return host, X, p


def verdict(p, thr):
    label = "Phishing" if p >= thr else "Not phishing"
    borderline = abs(p - thr) <= BORDERLINE_MARGIN
    return label, borderline


# ---- UI ----
st.set_page_config(page_title="Phishing Website Detector", page_icon="🛡️")
st.title("Phishing Website Detector")
st.caption("Checks a URL's host name with a trained machine learning model.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Run section 13 of the notebook and "
             "place phishing_model.joblib next to app.py.")
    st.stop()

art = load_artifact()
thr = art["threshold"]

with st.sidebar:
    st.subheader("Model")
    st.write(f"**{art['model_name']}**")
    st.write(f"Decision threshold: **{thr:.2f}**")
    st.write("A URL is flagged as phishing when its phishing probability is at or above the threshold.")
    st.subheader("Limits")
    st.write("- Only the host name is used. Path, query and scheme are ignored.")
    st.write("- Trained on only 820 legitimate domains, so short popular domains can score near the threshold.")
    st.write("- This is a coursework model. It is not a replacement for browser or security-vendor protection.")

tab_single, tab_batch = st.tabs(["Single URL", "Batch"])

with tab_single:
    example = st.selectbox("Examples", list(EXAMPLES), key="example")
    url = st.text_input("URL or domain", value=EXAMPLES[example], placeholder="e.g. example.com/login")
    if st.button("Check URL", type="primary") and url.strip():
        host, X, p = score(url, art)
        label, borderline = verdict(p, thr)
        (st.error if label == "Phishing" else st.success)(f"{label}")
        c1, c2 = st.columns(2)
        c1.metric("Phishing probability", f"{p:.1%}")
        c2.metric("Threshold", f"{thr:.0%}")
        if borderline:
            st.warning("This score is close to the threshold, so treat the result with caution.")
        st.write(f"Host analysed: `{host}`")
        with st.expander("Features used"):
            st.dataframe(X.T.rename(columns={0: "value"}), width="stretch")

with tab_batch:
    text = st.text_area("One URL or domain per line", height=160)
    if st.button("Check all") and text.strip():
        rows = []
        for u in [x.strip() for x in text.splitlines() if x.strip()]:
            host, _, p = score(u, art)
            label, borderline = verdict(p, thr)
            rows.append({"input": u, "host": host, "phishing_probability": round(p, 4),
                         "prediction": label, "borderline": borderline})
        out = pd.DataFrame(rows)
        st.dataframe(out, width="stretch")
        st.download_button("Download CSV", out.to_csv(index=False), "predictions.csv", "text/csv")
