from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- 1. MODEL CONFIGURATION ---
# We use a limit of 2048 tokens to fit within the 12GB VRAM of the RTX 3060
max_seq_length = 1536 
dtype = None # Auto-detect (RTX 3060 supports bfloat16)

print("Load model Med42...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/dian/Documents/Disertasi/mimic/med42_local",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True, # IMPORTANT: Compresses the model to fit in 12GB memory
)

# --- 2. LoRA CONFIGURATION (Specialist Notes) ---
print("Add LoRA adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # "Notebook" size (larger is smarter, but requires more memory)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout 0 is optimized for Unsloth
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Massively saves VRAM
    random_state = 3407,
)

# --- 3. LOADING & PREPARING DATASET ---
print("Load dataset_training.jsonl...")
# Ensuring ChatML format is read correctly
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

def formatting_prompts_func(examples):
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
    return {"text": texts}

# Reading the JSONL file we just created
dataset = load_dataset("json", data_files="dataset_training.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# --- 4. TRAINING PROCESS ---
print("Starting Fine-Tuning process...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 1, # Processes only 1 piece of data at a time
        gradient_accumulation_steps = 8, # Simulates batch size 8 (1x8)
        warmup_steps = 10,
        num_train_epochs = 1, # 1 full pass of 2,266 records is enough for a start
        learning_rate = 2e-4,
        fp16 = False,
        bf16 = True, # RTX 3060 (Ampere) supports bf16, which is more stable!
        logging_steps = 10, # Report progress every 10 steps
        optim = "adamw_8bit", # Memory-efficient optimizer
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# Execute!
trainer_stats = trainer.train()

# --- 5. SAVING RESULTS (GRADUATED) ---
print("Saving the training result model...")
# We only save the "Notebook" (LoRA) itself, which is very small (around a few hundred MB)
model.save_pretrained("lora_model_med42_medical")
tokenizer.save_pretrained("lora_model_med42_medical")

print("DONE! The model has been successfully trained and saved in the 'lora_model_med42_medical' folder.")