import os, traceback, re
import sys
sys.path.insert(0, r'd:\Coding Projects\ai-helpdesk')

# Minimal .env loader (avoids dependency on python-dotenv)
def load_env(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = re.match(r"([A-Za-z0-9_]+)=(.*)$", line)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            # strip surrounding quotes if present
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            os.environ.setdefault(k, v)

load_env(r'd:\Coding Projects\ai-helpdesk\.env')

from app.minimax_llm import MiniMaxLLM

api_base = os.environ.get('MINIMAX_API_BASE')
api_key = os.environ.get('MINIMAX_API_KEY')
print('MINIMAX_API_BASE=', api_base)
print('MINIMAX_API_KEY set=', bool(api_key))

try:
    llm = MiniMaxLLM(api_key=api_key, api_base=api_base, model='MiniMax-M2.7')
    resp = llm._complete('Say hi')
    print('Response text:', repr(resp.text))
    print('Raw response keys:', list(resp.raw.keys()) if isinstance(resp.raw, dict) else type(resp.raw))
except Exception as e:
    print('Exception during call:')
    traceback.print_exc()
