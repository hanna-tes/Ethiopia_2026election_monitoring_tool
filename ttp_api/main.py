# ttp_api/main.py
import os
import torch  #  ADDED: Required for device handling
import time
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Literal

# Config
HOST = os.getenv('TTP_API_HOST', '127.0.0.1')
PORT = int(os.getenv('TTP_API_PORT', '8002'))
MODEL_PATH = os.getenv('TTP_MODEL_PATH', '/Users/hannateshager/Ethiopia_2026election_monitoring_tool/model_cache/gemma-disarm-phase3-ttp')
API_KEY = os.getenv('TTP_API_KEY', 'ethiopia-ttp-dev-key')

# 🔧 FIX: Removed underscores to match 'global model, tokenizer' in functions
model = None
tokenizer = None

def get_model():
    global model, tokenizer
    if model is None:
        try:
            from unsloth import FastModel
            from transformers import AutoTokenizer
            print(f"🔄 Loading Gemma model from {MODEL_PATH}...")
            
            # 🔧 FIX: Use from_pretrained for inference loading if available, 
            # or stick to for_inference if that's your specific Unsloth version method.
            # 'for_inference' is often the standard for unsloth 2024+
            model = FastModel.for_inference(MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            print("✅ Model loaded!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e
            
    return model, tokenizer

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
    if isinstance(content, str): return content
    if isinstance(content, list):
        return '\n'.join(item.get('text','') if isinstance(item,dict) and item.get('type')=='text' else str(item) for item in content)
    return str(content)

def build_reply(messages, temperature=0.1, max_tokens=512):
    """Generate reply using the loaded Gemma model."""
    # Use global model/tokenizer
    global model, tokenizer
    
    if model is None or tokenizer is None:
        model, tokenizer = get_model()
    
    # Normalize messages
    normalized = [{'role': m.role, 'content': flatten_content(m.content)} for m in messages]
    prompt = tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 🔧 FIX: Safely get device and move inputs
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=max_tokens,
        temperature=float(temperature or 0.1),
        do_sample=(float(temperature or 0.1) > 0),
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode response
    input_len = inputs["input_ids"].shape[1]
    completion = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    
    return completion, int(input_len), int(output_ids.shape[1] - input_len)
    

# Middleware
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
    return {"status": "ok", "model": "gemma-disarm-phase3-ttp", "loaded": model is not None}

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

# Initialize FastAPI app
app = FastAPI(title='Gemma DISARM TTP API')

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT, log_level='info')
