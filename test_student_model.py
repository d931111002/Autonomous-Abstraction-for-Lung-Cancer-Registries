import pandas as pd
import json
import re
import time
from tqdm import tqdm
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, classification_report

# --- 1. CONFIGURATION ---
MODEL_PATH = "lora_model_med42_medical" 
DATA_FILE = "lung_dataset.csv"
GROUND_TRUTH_FILE = "Ground truth (33 records validated).xlsx" # Now using the Excel file
OUTPUT_FILE = "med42_test_results.csv"

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

def extract_simple_diagnosis(json_str):
    try:
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned_json = json_str[start_idx:end_idx+1]
        else:
            cleaned_json = json_str
            
        data = json.loads(cleaned_json)
        
        if data.get('level_1_malignancy_found') is False:
            return "Negative"
        if data.get('level_5_6_neuroendocrine_details', {}).get('is_present'):
            specific = data['level_5_6_neuroendocrine_details'].get('specific_type')
            if specific and specific != "null": return specific.lower()
            return "small cell carcinoma"
        if data.get('level_4_squamous_details', {}).get('is_present'):
            return "squamous cell carcinoma"
        if data.get('level_3_adenocarcinoma_details', {}).get('is_present'):
            return "adenocarcinoma"

        primary_types = data.get('level_2_primary_types', [])
        if primary_types and len(primary_types) > 0 and primary_types[0] != "null":
            return primary_types[0].lower()

        return "Malignancy Present"
    except Exception:
        return "ERROR"

def rescue_json_error(raw_json):
    raw_json = str(raw_json).lower()
    if '"level_1_malignancy_found": false' in raw_json or '"level_1_malignancy_found":false' in raw_json: return "negative"
    if "small cell" in raw_json: return "small cell carcinoma"
    elif "squamous" in raw_json: return "squamous cell carcinoma"
    elif "adenocarcinoma" in raw_json: return "adenocarcinoma"
    return "other/negative"

def main():
    print("Loading Trained Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    df_test = pd.read_csv(DATA_FILE)
    texts_to_process = df_test['ORGH_REPORT1'].dropna().tolist()
    
    print(f"Starting inference on {len(texts_to_process)} medical records...")
    results = []
    
    for text in tqdm(texts_to_process, desc="Parsing Text"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"INPUT TEXT:\n{text}"}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        
        start_infer = time.time()
        outputs = model.generate(input_ids=inputs, max_new_tokens=1024, use_cache=True)
        end_infer = time.time()
        
        response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
        simplified_diag = extract_simple_diagnosis(response)
        
        results.append({
            "Text": text,
            "Raw_JSON": response,
            "Predicted_Diagnosis": simplified_diag,
            "Inference_Time_sec": round(end_infer - start_infer, 2)
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[TIME RESULT] Average inference time: {df_results['Inference_Time_sec'].mean():.2f} seconds per document")

    # --- EVALUATION BLOCK ---
    try:
        df_gt = pd.read_excel(GROUND_TRUTH_FILE, names=['Text', 'Label'], header=0) 
        gt_mapping = {}
        for _, row in df_gt.iterrows():
            text_gt = str(row['Text']).strip()
            label_gt = str(row['Label']).strip().lower()
            
            if "adenocarcinoma" in label_gt: label_gt = "adenocarcinoma"
            elif "squamous" in label_gt: label_gt = "squamous cell carcinoma"
            elif "small cell" in label_gt: label_gt = "small cell carcinoma"
            elif "benign" in label_gt or "no malignancy" in label_gt or "not a lung" in label_gt: label_gt = "negative"
            else: label_gt = "other/negative" 
            gt_mapping[text_gt] = label_gt

        y_true, y_pred, rescued_count = [], [], 0
        for _, row in df_results.iterrows():
            text_key = str(row['Text']).strip()
            if text_key in gt_mapping:
                pred = str(row['Predicted_Diagnosis']).strip()
                if pred == "ERROR":
                    pred = rescue_json_error(row['Raw_JSON'])
                    rescued_count += 1
                else: pred = pred.lower()
                
                if "adenocarcinoma" in pred: pred = "adenocarcinoma"
                elif "squamous" in pred: pred = "squamous cell carcinoma"
                elif "small cell" in pred: pred = "small cell carcinoma"
                elif "negative" in pred: pred = "negative"
                else: pred = "other/negative"

                y_true.append(gt_mapping[text_key])
                y_pred.append(pred)

        if y_true:
            print("\n==================================================")
            print(f"[EVALUATION RESULTS]")
            print(f"- Records matched: {len(y_true)}")
            print(f"- Rescued formats: {rescued_count}")
            print(f"- Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
            print("==================================================")
            print(classification_report(y_true, y_pred, zero_division=0))
    except Exception as e:
        print(f"\nFailed to evaluate: {e}")

if __name__ == "__main__":
    main()