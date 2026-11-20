"""MoTVLM's stateless image API for contact sheets or independent frames."""

import asyncio
import base64
import json
from pathlib import Path
import urllib.error
import urllib.request
from uuid import uuid4

from prompts import motvlm_instructions, motvlm_observation, motvlm_question

MODEL_ID = 'motvlm'
MAX_NEW_TOKENS = 384
GENERATE_PATH = '/generate'
CONTACT_SHEET_LAYOUT = 'single_image'


def configured_url(override=None):
    if override:
        return override.rstrip('/')
    tasks = Path(__file__).resolve().parents[1] / 'image_eval' / 'tasks.json'
    return json.loads(tasks.read_text(encoding='utf-8'))['base_url'].rstrip('/')


class MoTVLM:
    def __init__(self, url, timeout, rules, schema, *, independent_frames=False):
        self.url = url
        self.timeout = timeout
        self.history = {}
        self.independent_frames = independent_frames
        self.max_images = 1
        self.instructions = motvlm_instructions(rules, schema, independent=independent_frames)

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
        if self.independent_frames:
            limit = detail.get('max_images')
            if type(limit) is not int or limit < 1:
                raise RuntimeError('MoTVLM did not report a valid image limit')
            self.max_images = limit
        return detail

    async def start(self, _cwd=None):
        episode_id = uuid4().hex
        self.history[episode_id] = []
        return episode_id

    def remember(self, episode_id, answer):
        # Only keep observations that already passed the output and frame-ID checks.
        self.history[episode_id].append(motvlm_observation(answer))

    async def observe(self, episode_id, text, image_input):
        history = None if self.independent_frames else self.history[episode_id]
        question = motvlm_question(self.instructions, text, history)
        paths = image_input if self.independent_frames else [image_input]
        images = ['data:image/jpeg;base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')
                  for path in paths]
        request = {'question': question, 'max_new_tokens': MAX_NEW_TOKENS}
        if not self.independent_frames:
            request['layout'] = CONTACT_SHEET_LAYOUT
        response = await asyncio.to_thread(self.request_json, GENERATE_PATH, {**request, 'images': images})
        details = {'request': {'method': 'POST', 'path': GENERATE_PATH, **request}}
        if not isinstance(response, dict):
            return uuid4().hex, response, details
        details.update({key: response[key] for key in ('model', 'input_token_count', 'generated_token_count',
                                                       'output_limit_reached') if key in response})
        details['server_elapsed_seconds'] = response.get('elapsed_seconds')
        return uuid4().hex, response.get('text'), details
