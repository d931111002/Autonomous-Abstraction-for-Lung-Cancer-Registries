import pandas as pd
import json
import re
import time
import torch
import torch.utils._pytree
from tqdm import tqdm

# --- 0. PYTORCH COMPATIBILITY PATCH ---
for i in range(1, 8):
    if not hasattr(torch, f'int{i}'):
        setattr(torch, f'int{i}', torch.int8)
    if not hasattr(torch, f'uint{i}'):
        setattr(torch, f'uint{i}', torch.uint8)

if not hasattr(torch.utils._pytree, 'register_constant'):
    torch.utils._pytree.register_constant = lambda x: x

from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, classification_report
from transformers import StoppingCriteria, StoppingCriteriaList

# --- 1. CONFIGURATION ---
MODEL_PATH = "lora_model_qwen_medical" 
DATA_FILE = "lung_dataset.csv"
OUTPUT_FILE = "qwen_test_results.csv"

# --- 2. CUSTOM STOPPING CRITERIA (SMART JSON COUNTER) ---
class StopOnProperJSON(StoppingCriteria):
    """
    Intelligently stops generation only when the outer JSON object is fully closed.
    It counts '{' and '}' to handle nested JSON structures correctly.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_output = self.tokenizer.decode(input_ids[0])
        
        # Isolate the assistant's reply to avoid counting prompt braces
        if "<|im_start|>assistant\n" in decoded_output:
            reply = decoded_output.split("<|im_start|>assistant\n")[-1]
        else:
            reply = decoded_output

        # Only trigger stop if there's at least one '{' and the brackets are balanced
        if "{" in reply:
            json_part = reply[reply.find("{"):]
            if json_part.count("{") > 0 and json_part.count("{") == json_part.count("}"):
                return True
        return False

# --- 3. SYSTEM PROMPT ---
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
c) UNCONFIRMED/SUSPECTED: The text only mentions "suspicious for" or "favoring" without a definitive diagnosis.

LEVEL 2: PRIMARY LINEAGES
Identify the main category:
- Non-Small Cell Lung Cancer (NSCLC)
- Small Cell Lung Cancer (SCLC)
- Other primary thoracic tumors

LEVEL 3: ADENOCARCINOMA SUB-CLASSIFICATION
If Adenocarcinoma, specify if it is:
- Pre-invasive (Adenocarcinoma in situ - AIS)
- Minimally invasive (MIA)
- Invasive (Identify predominant subtype: Lepidic, Acinar, Papillary, Solid, Micropapillary)
- Variants (Mucinous, Colloid, Fetal, Enteric)

LEVEL 4: SQUAMOUS CELL CARCINOMA SUB-CLASSIFICATION
If Squamous, identify variant (Keratinizing, Non-keratinizing, Basaloid).

LEVEL 5 & 6: NEUROENDOCRINE SUB-CLASSIFICATION
Differentiate between:
- High-grade (Small Cell Carcinoma, Large Cell Neuroendocrine Carcinoma)
- Low/Intermediate-grade (Typical Carcinoid, Atypical Carcinoid)

OUTPUT FORMAT:
You MUST output ONLY a valid JSON object. Do not add any markdown, explanation, or conversational text. Use the following schema:
{
  "level_1_malignancy_found": true/false,
  "level_2_primary_types": ["Type 1", "Type 2"],
  "level_3_adenocarcinoma_details": {
    "is_present": true/false,
    "specific_variant": "string or null",
    "predominant_subtype": "string or null"
  },
  "level_4_squamous_details": {
    "is_present": true/false,
    "subtype": "string or null"
  },
  "level_5_6_neuroendocrine_details": {
    "is_present": true/false,
    "specific_type": "string or null"
  },
  "reasoning_trace": "Brief explanation mapping the findings to the logic tiers"
}
"""

def extract_json_from_text(text):
    """Robust JSON extractor that handles nested braces perfectly."""
    try:
        start_idx = text.find('{')
        if start_idx == -1: return None
        
        brace_count = 0
        for i, char in enumerate(text[start_idx:]):
            if char == '{': brace_count += 1
            elif char == '}': brace_count -= 1
            
            if brace_count == 0:
                json_str = text[start_idx:start_idx+i+1]
                return json.loads(json_str)
        return None
    except Exception:
        return None

def extract_simple_diagnosis(json_data):
    """Parses the generated JSON to extract a simplified primary diagnosis."""
    if not json_data:
        return "ERROR"
    try:
        if not json_data.get("level_1_malignancy_found", False):
            return "Negative"
        
        primary_types = json_data.get("level_2_primary_types", [])
        if not primary_types:
            return "Unknown Malignancy"
            
        diagnoses = []
        for p_type in primary_types:
            p_type_lower = p_type.lower()
            if "adenocarcinoma" in p_type_lower:
                adeno_details = json_data.get("level_3_adenocarcinoma_details", {})
                if adeno_details and adeno_details.get("is_present"):
                    variant = adeno_details.get("specific_variant")
                    subtype = adeno_details.get("predominant_subtype")
                    detail = variant if variant else subtype
                    diagnoses.append(f"Adenocarcinoma ({detail})" if detail else "Adenocarcinoma")
                else:
                    diagnoses.append("Adenocarcinoma")
            elif "squamous" in p_type_lower:
                sq_details = json_data.get("level_4_squamous_details", {})
                if sq_details and sq_details.get("is_present"):
                    subtype = sq_details.get("subtype")
                    diagnoses.append(f"Squamous Cell Carcinoma ({subtype})" if subtype else "Squamous Cell Carcinoma")
                else:
                    diagnoses.append("Squamous Cell Carcinoma")
            elif "neuroendocrine" in p_type_lower or "small cell" in p_type_lower:
                 ne_details = json_data.get("level_5_6_neuroendocrine_details", {})
                 if ne_details and ne_details.get("is_present"):
                     spec_type = ne_details.get("specific_type")
                     diagnoses.append(spec_type if spec_type else p_type)
                 else:
                     diagnoses.append(p_type)
            else:
                diagnoses.append(p_type)
                
        return " + ".join(diagnoses)
    except Exception as e:
        print(f"Error extracting diagnosis: {e}")
        return "ERROR"

def main():
    print("--- 1. LOADING MODEL & TOKENIZER ---")
    max_seq_length = 2048
    dtype = None 

    # Load Model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # Initialize Smart Stopping Criteria
    stop_criteria = StoppingCriteriaList([StopOnProperJSON(tokenizer)])

    print(f"\n--- 2. LOADING DATASET ({DATA_FILE}) ---")
    try:
        df = pd.read_csv(DATA_FILE)
        texts = df['ORGH_REPORT1'].tolist()
        print(f"Successfully loaded {len(texts)} medical records.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("\n--- 3. STARTING INFERENCE ---")
    results = []

    for idx, text in enumerate(tqdm(texts, desc="Processing Reports")):
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nINPUT TEXT:\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        start_time = time.time()
        
        # Generation with Smart Stopping Criteria
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400, 
            use_cache=True, 
            temperature=0.01,
            stopping_criteria=stop_criteria
        )
        
        inference_time = time.time() - start_time
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        assistant_reply = decoded_output.split("<|im_start|>assistant\n")[-1].strip()
        
        parsed_json = extract_json_from_text(assistant_reply)
        diagnosis = extract_simple_diagnosis(parsed_json)

        results.append({
            "Text": text[:500], 
            "Raw_JSON": assistant_reply,
            "Predicted_Diagnosis": diagnosis,
            "Inference_Time_sec": round(inference_time, 2)
        })

    print("\n--- 4. SAVING RESULTS ---")
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"Results successfully saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()