import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import joblib
import dill
import re
import pickle
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dotenv import load_dotenv
import os
import nltk

load_dotenv()

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

API_KEY = os.getenv("API_KEY")

# --- Load models once globally ---
if "MEMM_MODEL" not in st.session_state:
    st.session_state.MEMM_MODEL = joblib.load("memm_model.joblib")
    st.session_state.MEMM_VEC = joblib.load("memm_vectorizer.joblib")
    st.session_state.MEMM_LE = joblib.load("memm_label_encoder.joblib")
    with open("crf_model.pkl", "rb") as f:
        st.session_state.CRF_MODEL = pickle.load(f)
    with open("hmm_model.dill", "rb") as f:
        st.session_state.HMM_MODEL = dill.load(f)

MEMM_MODEL = st.session_state.MEMM_MODEL
MEMM_VEC = st.session_state.MEMM_VEC
MEMM_LE = st.session_state.MEMM_LE
CRF_MODEL = st.session_state.CRF_MODEL
HMM_MODEL = st.session_state.HMM_MODEL

# --- CVE Data Functions ---
def fetch_cve_data(params):
    params = {
    "keywordSearch": "Linux",
    "pubStartDate": "2024-01-01T00:00:00.000Z",
    "pubEndDate": "2024-04-01T00:00:00.000Z"
    }
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    headers = {"apiKey": API_KEY}
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching CVE data: {response.status_code}")
        return None
    
def word2features_simple(sent, i):
    word = sent[i]
    features = {
        'word': word,
        'is_upper': word.isupper(),
        'is_title': word.istitle(),
        'is_digit': word.isdigit()
    }
    if i > 0:
        features.update({'-1:word': sent[i-1], '-1:is_title': sent[i-1].istitle()})
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        features.update({'+1:word': sent[i+1], '+1:is_title': sent[i+1].istitle()})
    else:
        features['EOS'] = True
    return features

def extract_entities(df, model):
    all_entities = []
    for _, row in df.iterrows():
        tokens = word_tokenize(row["Description"])
        if model == "MEMM":
            feats = [word2features_simple(tokens, i) for i in range(len(tokens))]
            pred_tags = MEMM_LE.inverse_transform(MEMM_MODEL.predict(MEMM_VEC.transform(feats)))
        elif model == "CRF":
            feats = [word2features_simple(tokens, i) for i in range(len(tokens))]
            pred_tags = CRF_MODEL.predict_single(feats)
        elif model == "HMM":
            tagged = HMM_MODEL.tag(tokens)
            pred_tags = [tag for _, tag in tagged]

        for token, tag in zip(tokens, pred_tags):
            if any(prefix in tag for prefix in ["OS", "TOOL", "TIME", "VULNAME", "IP", "MAL", "ACT", "URL"]):
                all_entities.append((token, tag))

    return pd.DataFrame(all_entities, columns=["Entity", "Tag"])

def parse_cve_data(data):
    if not data or "vulnerabilities" not in data:
        return pd.DataFrame()

    cve_list = []
    for item in data["vulnerabilities"]:
        cve = item.get("cve", {})
        cve_id = cve.get("id", "N/A")
        description = cve.get("descriptions", [{}])[0].get("value", "No description")
        severity = cve.get("metrics", {}).get("cvssMetricV31", [{}])[0].get("cvssData", {}).get("baseSeverity", "Unknown")
        score = cve.get("metrics", {}).get("cvssMetricV31", [{}])[0].get("cvssData", {}).get("baseScore", 0)

        cve_list.append({
            "CVE ID": cve_id,
            "Description": description,
            "Severity": severity,
            "CVSS Score": score
        })

    return pd.DataFrame(cve_list)

def extract_year(cve_id):
    match = re.match(r'CVE-(\\d{4})', cve_id)
    if match:
        return f'{match.group(1)}-01-01'
    return None

def contains_key_entity(desc, model_choice):
    tokens = word_tokenize(desc)
    if model_choice == "MEMM":
        feats = [word2features_simple(tokens, i) for i in range(len(tokens))]
        tags = MEMM_LE.inverse_transform(MEMM_MODEL.predict(MEMM_VEC.transform(feats)))
    elif model_choice == "CRF":
        feats = [word2features_simple(tokens, i) for i in range(len(tokens))]
        tags = CRF_MODEL.predict_single(feats)
    elif model_choice == "HMM":
        tagged = HMM_MODEL.tag(tokens)
        tags = [tag for _, tag in tagged]
    return any(any(prefix in tag for prefix in ["OS", "TOOL", "TIME", "VULNAME", "IP", "MAL", "ACT", "URL"]) for tag in tags)


# --- Source Data Functions ---
def fetch_source_data(params):
    base_url = "https://services.nvd.nist.gov/rest/json/source/2.0"
    headers = {"apiKey": API_KEY}
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching source data: {response.status_code}")
        return None

def parse_source_data(data):
    if not data or "sources" not in data:
        return pd.DataFrame()

    source_list = []
    for src in data["sources"]:
        source_list.append({
            "Name": src.get("name", "N/A"),
            "Identifier": src.get("sourceIdentifier", "N/A"),
            "Created": src.get("created", "N/A"),
            "Last Modified": src.get("lastModified", "N/A")
        })

    return pd.DataFrame(source_list)

# --- Streamlit UI ---
def main():
    st.title("NVD Explorer - CVE & Source API 2.0")

    tab1, tab2, tab3, tab4 = st.tabs(["CVE Search", "Data Sources", "NER Extractor", "NER Analysis"])

    # --- CVE Tab ---
    with tab1:
        st.sidebar.header("Filter Options")
        
        # ⏳ Load cached CVE DataFrame
        df = st.session_state.get("cve_df", pd.DataFrame())

        # Sidebar Filters
        cve_id = st.sidebar.text_input("CVE ID (optional)")
        keyword = st.sidebar.text_input("Keyword (e.g., Microsoft, Linux)")
        severity = st.sidebar.selectbox("CVSS Severity", ["", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
        start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("today"))
        end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

        if st.sidebar.button("Fetch CVE Data"):
            params = {}
            if cve_id:
                params["cveId"] = cve_id
            if keyword:
                params["keywordSearch"] = keyword
            if severity:
                params["cvssV3Severity"] = severity
            params["pubStartDate"] = f"{start_date}T00:00:00.000Z"
            params["pubEndDate"] = f"{end_date}T23:59:59.999Z"

            with st.spinner("Fetching CVE data..."):
                data = fetch_cve_data(params)
                df = parse_cve_data(data)
                st.session_state["cve_df"] = df

        # ✅ Always render results from session state
        if not df.empty:
            st.write("### CVE Results")
            st.dataframe(df)

            fig = px.histogram(df, x="Severity", title="Vulnerability Severity Distribution")
            st.plotly_chart(fig)

            st.write("### Top 10 Vulnerabilities by CVSS Score")
            top_vulns = df.sort_values(by="CVSS Score", ascending=False).head(10)
            st.dataframe(top_vulns)
        else:
            st.info("Click 'Fetch CVE Data' to begin.")
    # --- Source Tab ---
    with tab2:
        st.subheader("NVD Data Sources")
        source_id = st.text_input("Source Identifier (e.g., cve@mitre.org)", "")
        src_start = st.date_input("Modified Start Date", pd.to_datetime("today"), key="src_start")
        src_end = st.date_input("Modified End Date", pd.to_datetime("today"), key="src_end")
        results_per_page = st.number_input("Results per page", min_value=1, max_value=1000, value=20)
        start_index = st.number_input("Start index (pagination)", min_value=0, value=0)

        src_params = {
            "resultsPerPage": results_per_page,
            "startIndex": start_index
        }
        if source_id:
            src_params["sourceIdentifier"] = source_id
        elif src_start and src_end:
            src_params["lastModStartDate"] = f"{src_start}T00:00:00.000Z"
            src_params["lastModEndDate"] = f"{src_end}T23:59:59.999Z"

        if st.button("Fetch Source Data"):
            with st.spinner("Fetching source data..."):
                source_data = fetch_source_data(src_params)
                source_df = parse_source_data(source_data)

                if not source_df.empty:
                    st.write("### Data Sources")
                    st.dataframe(source_df)
                else:
                    st.warning("No source data found for the given criteria.")

    # --- NER Tab ---
    with tab3:
        st.subheader("Cybersecurity NER Extractor")

        model_choice = st.selectbox("Choose the NER model", ["MEMM", "CRF", "HMM"])

        key_entity_prefixes = ["OS", "TOOL", "TIME", "VULNAME", "IP", "PROT", "EMAIL", "URL", "MAL", "ACT"]

        if "cve_df" in st.session_state:
            df = st.session_state["cve_df"]
            selected_cve = st.selectbox("Select a CVE for NER", df["CVE ID"].tolist())
            description = df[df["CVE ID"] == selected_cve]["Description"].values[0]
            st.write(f"**Description:** {description}")

            with st.spinner("Extracting key entities..."):
                import joblib
                from nltk.tokenize import word_tokenize
                import dill
                import pickle

                # Load and prepare tokenizer
                tokens = word_tokenize(description)

                def word2features_simple(sent, i):
                    word = sent[i]
                    features = {
                        'word': word,
                        'is_upper': word.isupper(),
                        'is_title': word.istitle(),
                        'is_digit': word.isdigit()
                    }
                    if i > 0:
                        word1 = sent[i - 1]
                        features.update({
                            '-1:word': word1,
                            '-1:is_title': word1.istitle()
                        })
                    else:
                        features['BOS'] = True

                    if i < len(sent) - 1:
                        word1 = sent[i + 1]
                        features.update({
                            '+1:word': word1,
                            '+1:is_title': word1.istitle()
                        })
                    else:
                        features['EOS'] = True

                    return features

                if model_choice == "MEMM":
                    clf = joblib.load("memm_model.joblib")
                    vec = joblib.load("memm_vectorizer.joblib")
                    le = joblib.load("memm_label_encoder.joblib")

                    features = [word2features_simple(tokens, i) for i in range(len(tokens))]
                    X_vec = vec.transform(features)
                    pred_enc = clf.predict(X_vec)
                    pred_tags = le.inverse_transform(pred_enc)

                elif model_choice == "CRF":
                    with open("crf_model.pkl", "rb") as f:
                        crf = pickle.load(f)
                    features = [word2features_simple(tokens, i) for i in range(len(tokens))]
                    pred_tags = crf.predict_single(features)

                elif model_choice == "HMM":
                    with open("hmm_model.dill", "rb") as f:
                        hmm_model = dill.load(f)
                    tagged = hmm_model.tag(tokens)
                    tagged = hmm_model.tag(tokens)
                    pred_tags = [tag for _, tag in tagged]

                    # # DEBUG
                    # st.write("HMM Tagged Output:")
                    # st.write(tagged)
                    
                    pred_tags = [tag for _, tag in tagged]

                # Filter only key entities
                ner_result = pd.DataFrame({"Token": tokens, "Tag": pred_tags})
                filtered = ner_result[ner_result["Tag"].apply(lambda tag: any(key in tag for key in key_entity_prefixes))]

                # Map tag prefixes to human-readable descriptions
                tag_descriptions = {
                    "OS": "Operating System",
                    "TOOL": "Tool or Utility",
                    "TIME": "Timestamp or Time Reference",
                    "VULNAME": "Name of the Vulnerability",
                    "IP": "IP Address or Network Location",
                    "PROT": "Protocol",
                    "EMAIL": "Email Address",
                    "URL": "Web Link",
                    "MAL": "Malware Name or Type",
                    "ACT": "Malicious or Suspicious Action"
                }

                def get_description(tag):
                    for key in tag_descriptions:
                        if key in tag:
                            return tag_descriptions[key]
                    return "Other"

                filtered = filtered.copy()
                filtered["Explanation"] = filtered["Tag"].apply(get_description)
                filtered = filtered.drop_duplicates(subset=["Token", "Explanation"])
                
                st.write("### Key Extracted Entities for Analysis")
                st.dataframe(filtered[["Token", "Explanation"]].reset_index(drop=True), use_container_width=True)
        else:
            st.info("Please fetch CVE data first from the 'CVE Search' tab.")

    with tab4:
        st.subheader("NER-Based Entity Analysis from CVE Descriptions")
        model_choice = st.selectbox("Select NER model for analysis", ["MEMM", "CRF", "HMM"], key="ner_model_analysis")
        if "cve_df" in st.session_state and not st.session_state["cve_df"].empty:
            df = st.session_state["cve_df"]
            entity_df = extract_entities(df, model_choice)
            tag_filter = st.multiselect("Filter by Entity Type", options=sorted(set([tag.split("-")[-1] for tag in entity_df["Tag"]])), default=[])
            if tag_filter:
                entity_df = entity_df[entity_df["Tag"].apply(lambda t: t.split("-")[-1] in tag_filter)]
            if not entity_df.empty:
                st.write("### 📊 Top Mentioned Entities")
                st.bar_chart(entity_df["Entity"].value_counts().head(15))

                st.write("### 🧠 Entity Type Distribution")
                entity_types = entity_df["Tag"].apply(lambda x: x.split("-")[-1])
                st.plotly_chart(px.pie(values=entity_types.value_counts().values,
                                       names=entity_types.value_counts().index,
                                       title="Entity Type Breakdown"))
                
                st.write("### 🔠 WordCloud of Entities")
                wc = WordCloud(width=800, height=300, background_color='white')
                wc.generate_from_frequencies(entity_df["Entity"].value_counts())
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

                st.write("### 📄 Download Extracted Entities")
                st.download_button(
                    label="📥 Download as CSV",
                    data=entity_df.to_csv(index=False).encode("utf-8"),
                    file_name="ner_entities_from_cves.csv",
                    mime="text/csv"
                )

                st.write("### 📊 Compare Models Side-by-Side")
                model_comparison = {}
                for model in ["MEMM", "CRF", "HMM"]:
                    edf = extract_entities(df, model)
                    model_comparison[model] = len(edf)

                comp_df = pd.DataFrame(list(model_comparison.items()), columns=["Model", "# Entities Detected"])
                st.bar_chart(comp_df.set_index("Model"))
                
if __name__ == "__main__":
    main()