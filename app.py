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
    try:
        h = urlparse(u).netloc.split(":")[0].lower()
    except ValueError:  # malformed input such as "http://[bad"; treated as "no host found"
        return ""
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


def subdomain_note(X, label):
    """Warn when a 'Phishing' result may come from the subdomain, which the training data barely covers."""
    if label == "Phishing" and X["subdomain_count"].iloc[0] >= 1 and X["has_ip"].iloc[0] == 0:
        return ("This host has a subdomain. The training data had very few legitimate hosts with "
                "subdomains, so this model tends to over-flag them. Check the site's main domain "
                "before deciding.")
    return ""


# ---- "why this result" (what-if explanation) ----
# Typical legitimate value of each feature (median over the legitimate rows of the dataset).
LEGIT_REF = {"url_length": 16, "num_dots": 1, "has_ip": 0, "digits_count": 1, "num_hyphens": 0,
             "subdomain_count": 0, "entropy": 3.455, "num_labels": 2, "longest_label": 12,
             "longest_digit_run": 1, "digit_ratio": 0.067, "hyphen_ratio": 0.0, "vowel_ratio": 0.375,
             "has_susp_word": 0}
# +1: phishing hosts have higher values, -1: lower values (class averages in the dataset)
DIRECTION = {"url_length": 1, "num_dots": 1, "has_ip": 1, "digits_count": 1, "num_hyphens": 1,
             "subdomain_count": 1, "entropy": -1, "num_labels": 1, "longest_label": -1,
             "longest_digit_run": 1, "digit_ratio": 1, "hyphen_ratio": 1, "vowel_ratio": -1,
             "has_susp_word": 1}
MIN_EFFECT = 0.02  # ignore signals that change the score by less than 2 percentage points

REASON_TEXT = {
    "url_length": lambda v, r, t: f"The host is {int(v)} characters long (typical legitimate host: {int(r)})",
    "num_dots": lambda v, r, t: f"The host has {int(v)} dots (typical: {int(r)})",
    "has_ip": lambda v, r, t: "The host is an IP address, not a domain name",
    "digits_count": lambda v, r, t: f"The host has {int(v)} digits (typical: {int(r)})",
    "num_hyphens": lambda v, r, t: f"The host has {int(v)} hyphens (typical: {int(r)})",
    "subdomain_count": lambda v, r, t: f"The host has {int(v)} subdomain(s) (typical: {int(r)})",
    "entropy": lambda v, r, t: f"Low character variety (score {v:.2f}, typical {r:.2f})",
    "num_labels": lambda v, r, t: f"The host has {int(v)} dot-separated parts (typical: {int(r)})",
    "longest_label": lambda v, r, t: f"The longest part is only {int(v)} characters (typical: {int(r)})",
    "longest_digit_run": lambda v, r, t: f"There is a run of {int(v)} digits in a row (typical: {int(r)})",
    "digit_ratio": lambda v, r, t: f"{v:.0%} of the host is digits (typical: {r:.0%})",
    "hyphen_ratio": lambda v, r, t: f"{v:.0%} of the host is hyphens (typical: {r:.0%})",
    "vowel_ratio": lambda v, r, t: f"Only {v:.0%} of the letters are vowels (typical: {r:.0%})",
    "has_susp_word": lambda v, r, t: "The host contains a word such as login, secure or verify",
    "tld_freq": lambda v, r, t: f"The domain ending '.{t}' is not typical of legitimate hosts",
}


def explain(X, art, p, tld, top=3):
    """Which features push this host toward phishing. Start from a host where every feature has a
    typical legitimate value, then change one feature at a time to this host's value and see how much
    the phishing score rises. Returns [(text, rise), ...], biggest first."""
    ref = dict(LEGIT_REF)
    ref["tld_freq"] = art["tld_freq_map"].get("com", 0.0)  # .com is the most common legitimate ending
    def moves_toward_phishing(c):
        v = float(X[c].iloc[0])
        if c == "tld_freq":
            return v != ref[c]
        return (v - ref[c]) * DIRECTION[c] > 0 and abs(v - ref[c]) >= 0.25 * abs(ref[c])
    cols = [c for c in X.columns if c in ref and moves_toward_phishing(c)]
    if not cols:
        return []
    base = pd.DataFrame([{c: ref[c] for c in X.columns}])[list(X.columns)]
    variants = pd.concat([base] * (len(cols) + 1), ignore_index=True)
    for i, c in enumerate(cols):
        variants.loc[i + 1, c] = X[c].iloc[0]
    probs = art["model"].predict_proba(variants)[:, 1]
    found = [(float(probs[i + 1] - probs[0]), c) for i, c in enumerate(cols)]
    found = sorted([f for f in found if f[0] >= MIN_EFFECT], reverse=True)
    return [(REASON_TEXT[c](float(X[c].iloc[0]), ref[c], tld), d) for d, c in found[:top]]


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
        if not host:
            st.warning("Could not read a host name from this input. Try something like example.com.")
        else:
            label, borderline = verdict(p, thr)
            (st.error if label == "Phishing" else st.warning)(
                label if label == "Phishing" else "Not flagged as phishing")
            c1, c2 = st.columns(2)
            c1.metric("Phishing probability", f"{p:.1%}")
            c2.metric("Threshold", f"{thr:.0%}")
            if borderline:
                st.warning("This score is close to the threshold, so treat the result with caution.")
            note = subdomain_note(X, label)
            if note:
                st.info(note)
            if "." not in host:
                st.info("This host has no dot (no domain extension), so the model is not reliable for it.")
            st.write(f"Host analysed: `{host}`")
            why = explain(X, art, p, get_tld(host))
            with st.expander("Why this result?", expanded=(label == "Phishing")):
                if why:
                    st.write("Signals that raised the phishing score:" if label == "Phishing"
                             else "Signals that look like phishing, but not enough to flag this host:")
                    for text, _ in why:
                        st.write(f"- {text}")
                    st.caption("Estimated by starting from a typical legitimate host and changing one feature "
                               "at a time to this host's value. Several signals together can matter "
                               "more than any one alone.")
                else:
                    st.write("No strong phishing signals were found in this host name.")
            with st.expander("Features used"):
                st.dataframe(X.T.rename(columns={0: "value"}), width="stretch")

with tab_batch:
    text = st.text_area("One URL or domain per line", height=160)
    if st.button("Check all") and text.strip():
        rows = []
        for u in [x.strip() for x in text.splitlines() if x.strip()]:
            host, X, p = score(u, art)
            if not host:
                rows.append({"input": u, "host": "", "phishing_probability": None,
                             "prediction": "Invalid input", "borderline": False, "note": "", "main_reasons": ""})
                continue
            label, borderline = verdict(p, thr)
            if label != "Phishing":
                label = "Not flagged as phishing"
            rows.append({"input": u, "host": host, "phishing_probability": round(p, 4),
                         "prediction": label, "borderline": borderline,
                         "note": "has subdomain, may be over-flagged" if subdomain_note(X, label) else "",
                         "main_reasons": "; ".join(t for t, _ in explain(X, art, p, get_tld(host), top=2))
                         if label == "Phishing" else ""})
        out = pd.DataFrame(rows)
        st.dataframe(out, width="stretch")
        st.download_button("Download CSV", out.to_csv(index=False), "predictions.csv", "text/csv")
