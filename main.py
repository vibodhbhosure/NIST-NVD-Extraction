import streamlit as st
import requests
import pandas as pd
import plotly.express as px

def fetch_cve_data(params):
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None

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

def main():
    st.title("CVE Data Explorer - NVD API")
    
    st.sidebar.header("Filter Options")
    cve_id = st.sidebar.text_input("CVE ID (optional)")
    keyword = st.sidebar.text_input("Keyword (e.g., Microsoft, Linux)")
    severity = st.sidebar.selectbox("CVSS Severity", ["", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
    
    params = {}
    if cve_id:
        params["cveId"] = cve_id
    if keyword:
        params["keyword"] = keyword
    if severity:
        params["cvssV3Severity"] = severity
    params["pubStartDate"] = f"{start_date}T00:00:00.000"
    params["pubEndDate"] = f"{end_date}T23:59:59.999"
    
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching data..."):
            data = fetch_cve_data(params)
            df = parse_cve_data(data)
            
            if not df.empty:
                st.write("### CVE Results")
                st.dataframe(df)
                
                # Severity distribution chart
                fig = px.histogram(df, x="Severity", title="Vulnerability Severity Distribution")
                st.plotly_chart(fig)
                
                # Top vulnerabilities by CVSS Score
                st.write("### Top 10 Vulnerabilities by CVSS Score")
                top_vulns = df.sort_values(by="CVSS Score", ascending=False).head(10)
                st.dataframe(top_vulns)
            else:
                st.warning("No vulnerabilities found for the given criteria.")

if __name__ == "__main__":
    main()
