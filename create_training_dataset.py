import pandas as pd
import json
import os
import re

# --- CONFIGURATION ---
RESULTS_CSV = 'v4_final_lung_cancer_extraction_results.csv'
TEXT_CSV = 'mimic_lung_cancer_tiered_ready.csv'
OUTPUT_JSONL = 'dataset_training.jsonl'

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
""".strip()

def main():
    print("Loading CSV files...")
    if not os.path.exists(RESULTS_CSV) or not os.path.exists(TEXT_CSV):
        print("Error: Ensure both results CSV and MIMIC input CSV are in the same directory!")
        return

    df_results = pd.read_csv(RESULTS_CSV)
    df_text = pd.read_csv(TEXT_CSV)

    # --- CRITICAL FIX: DEDUPLICATION ---
    # Remove any duplicate hadm_id to prevent row multiplication during merge
    initial_count = len(df_results)
    df_results = df_results.drop_duplicates(subset=['hadm_id'], keep='last')
    df_text = df_text.drop_duplicates(subset=['hadm_id'], keep='first')
    
    if len(df_results) < initial_count:
        print(f"Removed {initial_count - len(df_results)} duplicate entries.")

    print("Merging AI output with the original text...")
    df_merged = pd.merge(df_results, df_text[['hadm_id', 'processed_text']], on='hadm_id', how='inner')

    valid_data = []
    error_count = 0

    print("Executing Quality Control & Formatting...")
    for index, row in df_merged.iterrows():
        raw_output = str(row['gemini_json_output'])
        
        # 1. Filter out network/API errors
        if "error" in raw_output.lower() or "MAX_RETRIES" in raw_output:
            error_count += 1
            continue
            
        # 2. Validate & Clean JSON
        try:
            clean_output = re.sub(r'```[a-zA-Z]*\n?', '', raw_output).strip()
            clean_output = clean_output.replace('```', '')
            parsed_json = json.loads(clean_output)
            
            # Handle list outputs
            if isinstance(parsed_json, list):
                if len(parsed_json) > 0:
                    parsed_json = parsed_json[0]
                else:
                    error_count += 1
                    continue
                    
            final_json_string = json.dumps(parsed_json)
            
        except json.JSONDecodeError:
            error_count += 1
            continue

        # 3. Construct ChatML format
        chatml_record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"INPUT TEXT:\n{row['processed_text']}"},
                {"role": "assistant", "content": final_json_string}
            ]
        }
        valid_data.append(chatml_record)

    print(f"Saving {len(valid_data)} clean records to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for record in valid_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("COMPLETED!")
    print(f"- Data successfully converted: {len(valid_data)} records")
    print(f"- Data errors/removed: {error_count} records")

if __name__ == "__main__":
    main()