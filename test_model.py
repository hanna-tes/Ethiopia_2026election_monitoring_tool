import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("=" * 60)
print("🔍 Starting Gemma LoRA Model Load Test...")
print("=" * 60)

# Paths to your local model files
base_model_path = "./model_cache/gemma-4-base"
adapter_path = "./model_cache/gemma-lora-hate-speech"

# 1. Load Tokenizer
print("\n[1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load Base Model (Using MPS for Mac GPU, or CPU as fallback)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[2/4] Loading base model onto {device.upper()}...")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map=device,
    low_cpu_mem_usage=True
)

# 3. Load LoRA Adapter
print("[3/4] Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("\n✅ Model loaded successfully! Running test inference...\n")

# 4. Define the Amharic Test Text
test_text = "የኦዴድ ንፈኛ ኦሮሙማ ወደር በደርዘን ተማርኳልይህ ሀር አፍራሽ ሰራዊት በሚባው ቋንቋ ገሩ አግባብ ሰለሆነ አራት ዲቃ በአንድ ቀን ማርኳል። በቅርቡ ሉ ለሉ ይበተናል️ ጋሽ አስታቄ እናመግናለን ¡"

categories = (
    "ethnicity, xenophobia, violence, extremism, gender disinformation, "
    "stereotype, class, derogatory, slur, misogynistic, religion, "
    "ethnic slur, inflammatory, inciteful, call for action, homophobic, "
    "structural, dehumanization, neutral"
)

# 🔥 CRITICAL FIX: Use the tokenizer's chat template!
# Instruct models (-it-) require specific formatting to know how to respond.
# Without this, they just output an empty EOS token.
messages = [
    {
        "role": "user", 
        "content": f"Classify the following text into one of these categories: {categories}\n\nText: \"{test_text}\"\n\nCategory:"
    }
]

# Apply the chat template to format the prompt correctly
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

print(f"📝 FORMATTED PROMPT:\n{prompt}\n")
print("-" * 60)

# 5. Tokenize the formatted prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("⏳ Generating prediction (this may take a few seconds)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,      # Give it enough room
        min_new_tokens=1,       # Force at least 1 token
        do_sample=True,         # 🔥 Enable sampling (greedy can get stuck on Instruct models)
        temperature=0.1,        # Low temperature for focused results
        repetition_penalty=1.2, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# 6. Decode and Debug
generated_ids = outputs[0][inputs['input_ids'].shape[1]:]

# 🔍 DEBUG: Print raw tokens to see exactly what the model generated
raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
print(f"🔍 Raw Decoded Text: '{raw_text}'")
print(f"🔍 Raw Token IDs: {generated_ids.tolist()}")

# Clean up for final display
prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
prediction = prediction.split('\n')[0].strip()
prediction = prediction.rstrip(':.,! ')

print(f"\n🤖 FINAL PREDICTION: '{prediction}'")
print("=" * 60)
print("✅ Test complete!")

