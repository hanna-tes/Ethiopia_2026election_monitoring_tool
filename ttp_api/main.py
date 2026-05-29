# ttp_api/main.py
import os
import sys
import json
import time
import uuid
import torch  # Required for device handling
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Literal

# Config
HOST = os.getenv('TTP_API_HOST', '127.0.0.1')
PORT = int(os.getenv('TTP_API_PORT', '8002'))
MODEL_PATH = os.getenv('TTP_MODEL_PATH', '/Users/hannateshager/Ethiopia_2026election_monitoring_tool/model_cache/gemma-disarm-phase3-ttp')
API_KEY = os.getenv('TTP_API_KEY', 'ethiopia-ttp-dev-key')

# Define app EARLY so middleware can use it
app = FastAPI(title='Gemma DISARM TTP API')

# Global model variables
_model = None
_tokenizer = None

def get_model():
    """Load Gemma base model (4-bit) + apply local LoRA adapter."""
    global _model, _tokenizer
    if _model is None:
        try:
            import os
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            print(f"🔄 Loading base model + applying local adapter from: {MODEL_PATH}...")
            
            # Verify adapter files exist
            required = ['adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json']
            missing = [f for f in required if not os.path.exists(os.path.join(MODEL_PATH, f))]
            if missing:
                raise FileNotFoundError(f"Missing adapter files: {missing}")
            
            # 🔧 FIX: Use a verified 4-bit model ID that definitely exists
            # Option A: Use Hugging Face's official 4-bit Gemma (if available)
            # Option B: Use any base model + quantize on-the-fly with BitsAndBytes
            base_model_name = "google/gemma-2b-it"  # Smaller, faster, reliable
            
            print(f"📦 Loading base model: {base_model_name} with 4-bit quantization...")
            
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load model with quantization
            _model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Reduce RAM usage during load
            )
            
            # Load tokenizer from your adapter folder (preserves your chat template)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            
            # Apply your fine-tuned LoRA adapter
            print(f"🔗 Applying LoRA adapter from {MODEL_PATH}...")
            _model = PeftModel.from_pretrained(_model, MODEL_PATH)
            _model.eval()
            
            print(f"✅ Model + adapter loaded successfully!")
            print(f"   Type: {type(_model)}")
            if hasattr(_model, 'device'):
                print(f"   Device: {_model.device}")
            print(f"   Memory usage: ~{sum(p.numel() * p.element_size() for p in _model.parameters()) / 1024**3:.2f} GB")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("💡 Quick fixes:")
            print("   1. pip install --upgrade bitsandbytes peft transformers accelerate")
            print("   2. Ensure you have ~8GB free RAM for 4-bit loading")
            print("   3. Try CPU-only: remove 'device_map' and 'torch.cuda' checks")
            import traceback
            traceback.print_exc()
            raise e
            
    return _model, _tokenizer

class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: Union[str, list]

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = 'gemma-disarm-phase3-ttp'
    messages: List[Message]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

def flatten_content(content):
    if isinstance(content, str): 
        return content
    if isinstance(content, list):
        return '\n'.join(item.get('text','') if isinstance(item,dict) and item.get('type')=='text' else str(item) for item in content)
    return str(content)

def build_reply(messages, temperature=0.1, max_tokens=512):
    """Generate reply using the loaded Gemma model."""
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        _model, _tokenizer = get_model()
    
    # Debug: Print what we're working with
    print(f"🔍 build_reply: model type={type(_model)}, tokenizer type={type(_tokenizer)}")
    
    # Normalize messages
    normalized = [{'role': m.role, 'content': flatten_content(m.content)} for m in messages]
    prompt = _tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
    
    # Tokenize input
    inputs = _tokenizer(prompt, return_tensors="pt")
    
    # Safely get device and move inputs
    if hasattr(_model, 'device'):
        device = _model.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    print(f"🤖 Generating with max_new_tokens={max_tokens}, temperature={temperature}...")
    with torch.inference_mode():
        output_ids = _model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=float(temperature or 0.1),
            do_sample=(float(temperature or 0.1) > 0),
            use_cache=True,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id
        )
    
    # Decode response
    input_len = inputs["input_ids"].shape[1]
    completion = _tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    
    print(f"✅ Generated {len(completion)} chars")
    return completion, int(input_len), int(output_ids.shape[1] - input_len)

@app.middleware("http")
async def auth_middleware(request, call_next):
    if request.url.path in ["/health", "/v1/models", "/openapi.json"]:
        return await call_next(request)
    key = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return await call_next(request)

@app.get("/health")
def health():
    return {"status": "ok", "model": "gemma-disarm-phase3-ttp", "loaded": _model is not None}

@app.get('/v1/models')
def list_models():
    return {'object': 'list', 'data': [{'id': 'gemma-disarm-phase3-ttp', 'object': 'model', 'owned_by': 'local'}]}

@app.post('/v1/chat/completions')
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail='Streaming not implemented')
    
    text, prompt_tokens, completion_tokens = build_reply(
        req.messages, temperature=req.temperature, max_tokens=req.max_tokens
    )
    
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex[:12]}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': req.model or 'gemma-disarm-phase3-ttp',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        }
    }

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT, log_level='info')
