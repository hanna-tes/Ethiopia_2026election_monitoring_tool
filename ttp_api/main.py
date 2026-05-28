# ttp_api/main.py
import os, sys, json, time, uuid
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Union, Literal
import uvicorn

# Config - matches your project structure
HOST = os.getenv('TTP_API_HOST', '127.0.0.1')
PORT = int(os.getenv('TTP_API_PORT', '8002'))
MODEL_PATH = os.getenv('TTP_MODEL_PATH', '/Users/hannateshager/Ethiopia_2026election_monitoring_tool/model_cache/gemma-disarm-phase3-ttp')
API_KEY = os.getenv('TTP_API_KEY', 'ethiopia-ttp-dev-key')

# Lazy-load model (loads once on first request)
_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        from unsloth import FastModel
        from transformers import AutoTokenizer
        print(f"🔄 Loading Gemma model from {MODEL_PATH}...")
        _model = FastModel.for_inference(MODEL_PATH)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("✅ Model loaded!")
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
    if isinstance(content, str): return content
    if isinstance(content, list):
        return '\n'.join(item.get('text','') if isinstance(item,dict) and item.get('type')=='text' else str(item) for item in content)
    return str(content)

def build_reply(messages, temperature=0.1, max_tokens=512):
    model, tokenizer = get_model()
    normalized = [{'role': m.role, 'content': flatten_content(m.content)} for m in messages]
    prompt = tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[prompt], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs, max_new_tokens=max_tokens,
        temperature=float(temperature or 0.1),
        do_sample=(float(temperature or 0.1) > 0),
        use_cache=True,
    )
    input_len = inputs["input_ids"].shape[1]
    completion = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return completion, int(input_len), int(output_ids.shape[1] - input_len)

app = FastAPI(title='Gemma DISARM TTP API')

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
