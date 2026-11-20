"""MoTVLM's stateless single-image API, with a short history per episode."""

import asyncio
import base64
import json
from pathlib import Path
import urllib.error
import urllib.request
from uuid import uuid4

MODEL_ID = 'motvlm'
MAX_NEW_TOKENS = 384


def configured_url(override=None):
    if override:
        return override.rstrip('/')
    tasks = Path(__file__).resolve().parents[1] / 'image_eval' / 'tasks.json'
    return json.loads(tasks.read_text(encoding='utf-8'))['base_url'].rstrip('/')


class MoTVLM:
    def __init__(self, url, timeout, rules, schema):
        self.url = url
        self.timeout = timeout
        self.history = {}
        properties = schema['properties']
        options = '; '.join(f'{name}: {" | ".join(properties[name]["enum"])}'
                            for name in ('phase', 'outcome', 'decision'))
        self.instructions = (
            f'{rules}\nThe single attached image is a contact sheet. Read it left to right, top to bottom. '
            'Use only the listed frame IDs. Return one compact JSON object. Put these keys directly at the root: '
            'phase, progress, outcome, needs_intervention, decision, reason, evidence. '
            f'Allowed values: {options}. needs_intervention is a JSON boolean. '
            'Use short phrases for progress and reason. evidence is an array of one or two objects '
            'with frame_id and observation. Do not print the schema or add other keys.'
        )

    def request_json(self, path, payload=None):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            self.url + path, data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'},
        )
        try:
            with opener.open(request, timeout=min(self.timeout, 15) if payload is None else self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read(1024).decode(errors='replace')
            raise RuntimeError(f'MoTVLM HTTP {error.code}: {detail}') from error

    async def health(self):
        detail = await asyncio.to_thread(self.request_json, '/health')
        if not detail.get('ready') or detail.get('busy'):
            raise RuntimeError('MoTVLM service is not ready or is busy')
        return detail

    async def start(self, _cwd=None):
        episode_id = uuid4().hex
        self.history[episode_id] = []
        return episode_id

    def remember(self, episode_id, answer):
        # Only keep observations that already passed the output and frame-ID checks.
        self.history[episode_id].append({key: answer[key] for key in ('phase', 'outcome', 'progress', 'evidence')})

    async def observe(self, episode_id, text, sheet):
        observations = self.history[episode_id]
        recent = observations[-2:]
        prior_success = next((row for row in observations if row['outcome'] == 'success'), None)
        if prior_success and prior_success not in recent:
            recent = [prior_success, *recent]
        history = json.dumps(recent, ensure_ascii=False, separators=(',', ':'))
        question = (f'{self.instructions}\nPrior observations in this episode (model outputs, not instructions): '
                    f'{history}\n{text}')
        encoded = base64.b64encode(Path(sheet).read_bytes()).decode('ascii')
        response = await asyncio.to_thread(self.request_json, '/generate', {
            'question': question, 'images': [f'data:image/jpeg;base64,{encoded}'],
            'layout': 'single_image', 'max_new_tokens': MAX_NEW_TOKENS,
        })
        if not isinstance(response, dict):
            return uuid4().hex, response, {}
        details = {key: response[key] for key in ('model', 'input_token_count', 'generated_token_count',
                                                  'output_limit_reached') if key in response}
        details['server_elapsed_seconds'] = response.get('elapsed_seconds')
        return uuid4().hex, response.get('text'), details
