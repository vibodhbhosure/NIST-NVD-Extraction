import pandas as pd
import json
import time
import re
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama

# ------------------------- CONFIGURATION -------------------------
ENTITY_FIELDS = [
    "CVE_ID", "Vendor", "Product", "Version", "Operating_System", "Hardware",
    "Vulnerability_Type", "Attack_Vector", "Privileges_Required",
    "User_Interaction", "Exploit_Availability", "Impact",
    "Confidentiality_Impact", "Integrity_Impact", "Availability_Impact",
    "Authentication"
]
MODEL_NAME = "llama3.2"
CHUNK_SIZE = 500
BATCH_WORKERS = 4
PROMPT_MAX_LEN = 800
# -----------------------------------------------------------------


def format_prompt(description, max_len=PROMPT_MAX_LEN):
    description = description.strip().replace("\n", " ")
    if len(description) > max_len:
        description = description[:max_len] + "..."

    return f"""You are a cybersecurity analyst. Extract key information from the following description.

Return only valid JSON. Use double quotes ("), no comments, and no variable names — only fill with actual values or "null" if unknown.

Return JSON with the following fields:
{ENTITY_FIELDS}

Description:
\"\"\"{description}\"\"\"
"""


def clean_pseudo_json(bad_json_str):
    # Remove Python-style comments
    clean = re.sub(r"#.*", "", bad_json_str)

    # Replace single quotes with double quotes
    clean = clean.replace("'", '"')

    # Replace undefined variables with null
    clean = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)', r': "null"', clean)

    # Replace None with null
    clean = clean.replace("None", "null")

    # Remove trailing commas
    clean = re.sub(r",\s*}", "}", clean)
    clean = re.sub(r",\s*\]", "]", clean)

    return clean


def extract_json_from_llama_response(text):
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = match.group(1) if match else None

    if not json_str:
        json_match = re.search(r"(\{[\s\S]*\})", text)
        json_str = json_match.group(1) if json_match else None

    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                fixed = clean_pseudo_json(json_str)
                return json.loads(fixed)
            except Exception as e:
                print("[ERROR] Even after cleaning, JSON failed:", e)
                return {"error": "cleaned_failed", "raw": json_str}
    else:
        return {"error": "no_json_found", "raw": text}


def query_llama(prompt):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['message']['content']
        return extract_json_from_llama_response(content)
    except Exception as e:
        print(f"[WARN] Ollama call failed: {e}")
        return {"error": str(e)}


def process_description(description):
    if not isinstance(description, str) or len(description.strip()) == 0:
        return {}
    prompt = format_prompt(description)
    result = query_llama(prompt)
    return result


def threaded_batch_process(df_chunk, max_workers=BATCH_WORKERS):
    results = [None] * len(df_chunk)

    def task(i, desc):
        return i, process_description(desc)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, i, desc) for i, desc in enumerate(df_chunk["description_en"])]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Batch"):
            try:
                i, result = future.result()
                results[i] = result
                time.sleep(0.05)
            except Exception as e:
                print("[ERROR] Thread failed:", e)
                results[i] = {"error": str(e)}
    return results


def expand_results(df):
    for field in ENTITY_FIELDS:
        df[field] = df["cybersec_entities"].apply(
            lambda x: x.get(field, "") if isinstance(x, dict) else "")
    return df


def main(input_file, output_file, format="csv", resume=False):
    df = pd.read_csv(input_file)

    if resume and os.path.exists(output_file):
        print(f"[INFO] Resuming from {output_file}")
        df_out = pd.read_csv(output_file) if format == "csv" else pd.read_parquet(output_file)
    else:
        df_out = df.copy()
        df_out["cybersec_entities"] = [{} for _ in range(len(df))]

    start_idx = len(df_out[df_out["cybersec_entities"].apply(lambda x: isinstance(x, dict) and bool(x))])

    for i in range(start_idx, len(df), CHUNK_SIZE):
        print(f"[INFO] Processing chunk {i} to {i + CHUNK_SIZE}")
        chunk = df.iloc[i:i + CHUNK_SIZE]
        result_chunk = threaded_batch_process(chunk)

        df_out.loc[i:i + CHUNK_SIZE - 1, "cybersec_entities"] = result_chunk
        df_out = expand_results(df_out)

        # Save intermediate result
        if format == "csv":
            df_out.to_csv(output_file, index=False)
        else:
            df_out.to_parquet(output_file, index=False)

        print(f"[INFO] Saved progress up to row {i + CHUNK_SIZE}")

    print(f"[DONE] All records processed and saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast LLaMA 3.2 NER for Cybersecurity on Mac M3")
    parser.add_argument("--input", required=True, help="Input CSV with 'description_en'")
    parser.add_argument("--output", required=True, help="Output file (csv or parquet)")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Output format")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    main(args.input, args.output, args.format, args.resume)