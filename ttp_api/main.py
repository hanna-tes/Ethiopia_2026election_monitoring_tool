# ttp_api/main.py
import os
import json
import time
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import uvicorn

# MLX Native Loading (Much faster on Apple Silicon)
from mlx_lm import load, generate

logger = logging.getLogger(__name__)

# Configuration
HOST = os.getenv('TTP_API_HOST', '127.0.0.1')
PORT = int(os.getenv('TTP_API_PORT', '8002'))
# Point to the NEW fused model we just created in Step 1
MODEL_PATH = '/Users/hannateshager/Ethiopia_2026election_monitoring_tool/model_cache/gemma-merged'

app = FastAPI(title='Gemma DISARM TTP API (MLX Native)')

# Global model state
_model = None
_tokenizer = None

def get_model():
    """Load the fused MLX model once and cache it."""
    global _model, _tokenizer
    if _model is None:
        logger.info(f"🔄 Loading fused MLX model from {MODEL_PATH}...")
        try:
            # mlx_lm.load returns both model and tokenizer
            _model, _tokenizer = load(MODEL_PATH)
            logger.info("✅ Model loaded successfully via MLX!")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e
    return _model, _tokenizer

class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.1
    max_tokens: int = 1024

def build_reply(messages, temperature=0.1, max_tokens=1024):
    """Generate response using native MLX generation."""
    model, tokenizer = get_model()
    
    # Format messages into a single prompt string
    prompt_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Generate using MLX (native Apple Silicon acceleration)
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temp=temperature,
    )
    
    return response

@app.post('/v1/chat/completions')
def chat_completions(req: ChatCompletionRequest):
    try:
        logger.info(" Generating TTP analysis with MLX...")
        
        # Generate the response
        text = build_reply(req.messages, req.temperature, req.max_tokens)
        
        # Estimate token usage (rough approximation for MLX)
        prompt_tokens = sum(len(m.content.split()) * 1.3 for m in req.messages)
        completion_tokens = len(text.split()) * 1.3
        
        return {
            'id': f'chatcmpl-{uuid.uuid4().hex[:12]}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': 'gemma-disarm-phase3-ttp-mlx',
            'choices': [{
                'index': 0, 
                'message': {'role': 'assistant', 'content': text}, 
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': int(prompt_tokens), 
                'completion_tokens': int(completion_tokens), 
                'total_tokens': int(prompt_tokens + completion_tokens)
            }
        }
    except Exception as e:
        logger.error(f"❌ Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health(): 
    return {"status": "ok", "backend": "mlx_lm"}

@app.get('/v1/models')
def list_models(): 
    return {'object': 'list', 'data': [{'id': 'gemma-disarm-phase3-ttp-mlx', 'object': 'model'}]}

if __name__ == '__main__':
    logger.info(f"🚀 Starting MLX TTP API on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
