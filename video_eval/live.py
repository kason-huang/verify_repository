"""Schedule fresh model calls from the browser's video clock."""

import asyncio
import json
import time
from copy import deepcopy
from threading import Condition, Thread
from uuid import uuid4

from reporting import asset_url
from run import WARMUP_PROMPT, observe_window, open_astra, warmup_thread

IDLE_SECONDS = 180


class LiveRun:
    def __init__(self, episode, model, output, warmup=False):
        self.id = uuid4().hex
        self.episode = episode
        self.model = model
        self.output = output
        self.warmup_requested = warmup
        self.warmup_started_at = None
        self.warmup = None
        self.condition = Condition()
        self.ready = 0
        self.queued_at = []
        self.playhead = 0.0
        self.rows = []
        self.in_flight = None
        self.started_at = None
        self.status = 'connecting'
        self.error = None
        self.stopping = False
        self.last_advance = time.monotonic()
        self.directory = output / 'live-runs' / self.id
        self.directory.mkdir(parents=True)
        self.write_config()
        self.worker = Thread(target=self.run, daemon=True)
        self.worker.start()

    def write_config(self):
        episode = self.episode
        metadata = {'id': self.id, 'episode': episode['episode'] if episode else None, 'model': self.model,
                    'video_sha256': episode['sha256'] if episode else None,
                    'warmup_prompt': WARMUP_PROMPT if self.warmup_requested else None}
        (self.directory / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')

    def select_episode(self, episode):
        with self.condition:
            if self.status != 'ready' or self.episode is not None or self.stopping:
                raise ValueError('这个会话还未就绪，或已选定视频。')
            self.episode = episode
            self.write_config()
            self.last_advance = time.monotonic()
            self.status = 'waiting'
            self.condition.notify_all()

    def advance(self, playhead, ended):
        with self.condition:
            if self.episode is None:
                raise ValueError('请先选择视频并开始监控。')
            if self.warmup_requested and self.warmup is None:
                raise ValueError('请等模型预热完成再开始播放。')
            self.last_advance = time.monotonic()
            self.playhead = max(playhead, self.playhead)
            if self.status == 'ready':
                self.status = 'waiting'
            eligible = sum(window['observed_at_s'] <= self.playhead + 0.001
                           and (not window['end_of_episode'] or ended)
                           for window in self.episode['windows'])
            if eligible > self.ready:
                self.queued_at.extend([self.last_advance] * (eligible - self.ready))
                self.ready = eligible
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            rows = deepcopy(self.rows)
            for row in rows:
                row['sheet'] = asset_url(self.output, row['sheet'], live=True)
                for evidence in row.get('answer', {}).get('evidence', []):
                    evidence['path'] = asset_url(self.output, evidence['path'], live=True)
                row.pop('raw_response', None)
                row.pop('thread_id', None)
                row.pop('turn_id', None)
            return {
                'id': self.id, 'model': self.model, 'episode': self.episode['episode'] if self.episode else None,
                'playhead_s': self.playhead,
                'status': 'stopping' if self.stopping and self.status not in {'stopped', 'error', 'complete'} else self.status,
                'waiting_count': self.ready - len(rows) - int(self.in_flight is not None),
                'in_flight_s': self.in_flight,
                'elapsed_s': round(time.monotonic() - self.started_at, 2) if self.started_at is not None else None,
                'warmup_requested': self.warmup_requested,
                'warmup_elapsed_s': (round(time.monotonic() - self.warmup_started_at, 2)
                                     if self.warmup_started_at is not None else None),
                'warmup_seconds': self.warmup['turn_completed_seconds'] if self.warmup else None,
                'timeline': rows, 'error': self.error,
            }

    def wait_for_input(self, available):
        while not available() and not self.stopping:
            idle = IDLE_SECONDS - (time.monotonic() - self.last_advance)
            if idle <= 0:
                self.stopping = True
                break
            self.condition.wait(timeout=idle)
        return not self.stopping

    def wait_for_episode(self):
        with self.condition:
            return self.episode if self.wait_for_input(lambda: self.episode is not None) else None

    def wait_for_window(self, index):
        with self.condition:
            if not self.wait_for_input(lambda: self.ready > index):
                return False
            self.status = 'inferencing'
            self.in_flight = self.episode['windows'][index]['observed_at_s']
            self.started_at = time.monotonic()
            return True

    async def monitor(self):
        async with open_astra(self.directory / 'codex.log', timeout=120, default_model=self.model) as (astra, cwd):
            thread = await astra.start(cwd, self.model)
            with self.condition:
                if self.stopping:
                    return
                self.status = 'warming' if self.warmup_requested else 'waiting'
                if self.warmup_requested:
                    self.warmup_started_at = time.monotonic()
            if self.warmup_requested:
                info = await warmup_thread(astra, thread)
                (self.directory / 'warmup.json').write_text(json.dumps({'thread_id': thread, **info}, indent=2) + '\n')
                with self.condition:
                    self.warmup = info
                    self.warmup_started_at = None
                    self.last_advance = time.monotonic()
                    self.status = 'ready'
            episode = await asyncio.to_thread(self.wait_for_episode)
            if episode is None:
                return
            known = set()
            with (self.directory / 'timeline.jsonl').open('w', encoding='utf-8') as log:
                for index, window in enumerate(episode['windows']):
                    if not await asyncio.to_thread(self.wait_for_window, index):
                        break
                    row = await observe_window(astra, thread, episode, window, known)
                    with self.condition:
                        row['queue_seconds'] = round(self.started_at - self.queued_at[index], 3)
                        row['playhead_at_reply_s'] = self.playhead
                        log.write(json.dumps(row, ensure_ascii=False) + '\n')
                        log.flush()
                        self.rows.append(row)
                        self.in_flight = self.started_at = None
                        self.status = 'waiting'
                    if 'error' in row:
                        raise RuntimeError(row['error'])

    def run(self):
        try:
            asyncio.run(self.monitor())
        except Exception as error:
            with self.condition:
                self.error = str(error) or type(error).__name__
                self.in_flight = self.started_at = self.warmup_started_at = None
                self.status = 'error'
        else:
            with self.condition:
                self.status = 'stopped' if self.stopping else 'complete'
