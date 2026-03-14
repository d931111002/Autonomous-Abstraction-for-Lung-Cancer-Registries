import pandas as pd
import google.generativeai as genai
import time
import json
import os
import re
from tqdm import tqdm

# --- CONFIGURATION ---
# REPLACE WITH YOUR API KEY
API_KEY = "AIzaSyA2Au4IsJhLnYL3RHc7JXPomngy43uzoU8" 

INPUT_FILE = 'mimic_lung_cancer_tiered_ready.csv'
OUTPUT_FILE = 'v4_final_lung_cancer_extraction_results.csv'

# Using Gemini 2.0 Flash (Stable, fast, and proven to be less prone to header-hallucinations)
MODEL_NAME = 'models/gemini-2.0-flash' 

# Configure Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a specialized AI Pathologist and Clinical Data Abstractor. 
YOUR OBJECTIVE:
Extract structured histological data from pathology reports strictly following the "WHO Classification of Thoracic Tumours (2021)" Tiered Logic. 

GENERALIZABILITY & SAFETY RULES:
1. This logic applies to any text input. Ignore formatting inconsistencies.
2. HEADER BLINDNESS: The input text contains automated extraction headers (e.g., "--- CONTEXT: MIA ---", "--- CONTEXT: ADENOCARCINOMA ---"). These are artifacts. You MUST IGNORE these headers. Base your diagnosis ONLY on the actual medical narrative and pathology results.

--- TIERED LOGIC INSTRUCTIONS ---
LEVEL 1: INITIAL ASSESSMENT (THE GATEKEEPER)
You MUST output "Negative" (and stop processing) if ANY of the following apply:
a) No tumor or malignancy is described.
b) NON-THORACIC PRIMARY: The primary tumor is from another organ (e.g., pancreatic, colon, breast), even if the word "adenocarcinoma" is present.
c) UNCONFIRMED/SUSPECTED: The text only mentions "lung nodules", "concerning for malignancy", or "suspected" WITHOUT explicit pathological or cytological confirmation.

If it is a confirmed primary thoracic/lung malignancy, PROCEED to LEVEL 2.

LEVEL 2: PRIMARY CLASSIFICATION
- Search for Adenocarcinoma, Squamous, Neuroendocrine, etc.
LEVEL 3: ADENOCARCINOMA DRILL-DOWN
- Check for AIS, MIA, Invasive Non-mucinous (Lepidic, Acinar, Papillary, Solid).
LEVEL 4: SQUAMOUS CELL DRILL-DOWN
- Keratinizing, Non-keratinizing, Basaloid.
LEVEL 5 & 6: NEUROENDOCRINE
- Small cell vs Large cell.

--- OUTPUT FORMAT ---
Return a valid JSON object ONLY. 
{
  "level_1_malignancy_found": boolean,
  "level_2_primary_types": ["List", "of", "types"],
  "level_3_adenocarcinoma_details": {
    "is_present": boolean,
    "specific_variant": "String or null",
    "predominant_subtype": "String or null"
  },
  "level_4_squamous_details": {
    "is_present": boolean,
    "subtype": "String or null"
  },
  "level_5_6_neuroendocrine_details": {
    "is_present": boolean,
    "specific_type": "String or null"
  },
  "reasoning_trace": "Brief explanation of your logic, citing specific medical evidence from the text."
}
"""

def extract_simple_diagnosis(json_str):
    """
    Extracts a simplified diagnosis string from the JSON output.
    Robustly handles lists and API errors.
    """
    try:
        cleaned_json = re.sub(r'```[a-zA-Z]*\n?', '', str(json_str)).strip()
        cleaned_json = cleaned_json.replace('```', '')
        
        data = json.loads(cleaned_json)
        
        if isinstance(data, list):
            if len(data) > 0:
                data = data[0]
            else:
                return "JSON_PARSE_ERROR: Empty List"
        
        if "error" in data:
            return f"API_ERROR: {data['error']}"

        if data.get('level_1_malignancy_found') is False:
            return "No Malignancy / Negative"

        if data.get('level_5_6_neuroendocrine_details', {}).get('is_present'):
            specific = data['level_5_6_neuroendocrine_details'].get('specific_type')
            if specific: return specific
            return "Neuroendocrine Carcinoma/Tumor"

        if data.get('level_4_squamous_details', {}).get('is_present'):
            subtype = data['level_4_squamous_details'].get('subtype')
            if subtype and subtype != "null": return f"Squamous cell carcinoma, {subtype}"
            return "Squamous cell carcinoma"

        if data.get('level_3_adenocarcinoma_details', {}).get('is_present'):
            variant = data['level_3_adenocarcinoma_details'].get('specific_variant')
            subtype = data['level_3_adenocarcinoma_details'].get('predominant_subtype')
            
            if variant and variant != "null": return f"Adenocarcinoma ({variant})"
            if subtype and subtype != "null": return f"Adenocarcinoma ({subtype} predominant)"
            return "Adenocarcinoma"

        primary_types = data.get('level_2_primary_types', [])
        if primary_types:
            valid_types = [t for t in primary_types if t]
            return ", ".join(valid_types)

        if data.get('level_1_malignancy_found'):
             return "Malignancy Present (Type Unspecified)"

        return "Indeterminate"

    except json.JSONDecodeError:
        return "JSON_PARSE_ERROR"
    except Exception as e:
        return f"UNKNOWN_ERROR: {str(e)}"

def get_gemini_response_with_retry(text, max_retries=10):
    wait_time = 4 
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                f"{SYSTEM_PROMPT}\n\nINPUT TEXT:\n{text}",
                generation_config={"response_mime_type": "application/json"}
            )
            return response.text
        except Exception as e:
            error_msg = str(e)
            print(f"\n[WARNING] Attempt {attempt+1}/{max_retries} failed: {error_msg[:100]}...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60) 
    
    return json.dumps({"error": "MAX_RETRIES_EXCEEDED"})

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found!")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} patient records.")

    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            if 'hadm_id' in df_existing.columns:
                processed_ids = df_existing['hadm_id'].tolist()
                print(f"Found {len(processed_ids)} previously processed records. Resuming...")
            else:
                processed_ids = []
        except pd.errors.EmptyDataError:
            processed_ids = []
    else:
        processed_ids = []
        header_df = pd.DataFrame(columns=['subject_id', 'hadm_id', 'gemini_json_output', 'simplified_diagnosis'])
        header_df.to_csv(OUTPUT_FILE, index=False)

    data_to_process = df[~df['hadm_id'].isin(processed_ids)]
    results_buffer = []
    
    for index, row in tqdm(data_to_process.iterrows(), total=data_to_process.shape[0], desc="Processing Gemini"):
        json_result = get_gemini_response_with_retry(row['processed_text'])
        simple_diag = extract_simple_diagnosis(json_result)
        
        results_buffer.append({
            'subject_id': row['subject_id'],
            'hadm_id': row['hadm_id'],
            'gemini_json_output': json_result,
            'simplified_diagnosis': simple_diag 
        })

        time.sleep(2) 

        if len(results_buffer) >= 5:
            temp_df = pd.DataFrame(results_buffer)
            temp_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            results_buffer = [] 

    if results_buffer:
        temp_df = pd.DataFrame(results_buffer)
        temp_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    print("COMPLETED! All records processed.")

if __name__ == "__main__":
    main()