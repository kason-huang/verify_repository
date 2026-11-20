#!/usr/bin/env python3
"""Replay the five LeRobot cam_high episodes to Astra or MoTVLM."""

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from prompts import window_prompt
from reporting import write_report
from responses import OUTPUT_SCHEMA, parse_answer, validate
from motvlm import MODEL_ID as MOTVLM_MODEL, MoTVLM, configured_url

ROOT = Path(__file__).resolve().parents[1]
TASK = 'use the right arm to pick up one box'
MODEL = 'gpt-6-astra'
EFFORT = 'low'
WINDOW_SECONDS = 3
SAMPLE_EVERY_FRAMES = 8
WARMUP_PROMPT = 'Connection check only. No video is provided. Reply with READY only and do not count this as a task observation.'
RULES = '''Evaluate the task instruction exactly as written. This is a causal replay of one
robot episode. Use the current chronological contact sheet and the observations from
this conversation only. Do not use tools. Judge phases from visible changes over time.
Use the robot's own right arm, and distinguish its side from the viewer's side.
Do not infer contact or lift just from a closed gripper. Success requires visual evidence
that the right arm grasped one box and lifted it clear of its original supporting surface.
Later placement is outside the task. Once visibly confirmed, retain success unless later
evidence shows the original observation was mistaken. A recoverable unsuccessful grasp
is an attempt failure, not an episode failure. Use `failure` on a clear terminal failed
attempt or at the end if the goal was visibly not achieved. Use `uncertain` if the available
views cannot verify the outcome; do not assume success from the episode ending.
Recommend `intervene` only for a blocked task, a wrong-arm lift, a dropped object visibly
out of reach, three clearly failed grasps, or at least six seconds without progress after a
visible failed attempt. These are evaluation rules, not facts about the robot. Ordinary
slow approach is not a stall. Recommend `retry` for a reversible failed attempt when
appropriate; otherwise use `continue`, `finish` on success, or `observe` if evidence is lacking.
Map `phase` to the latest meaningful task stage. `progress` describes the stage and what
changed since the previous check. `reason` explains the verdict and recommendation.
`evidence` cites one or two frame IDs previously supplied, with the visible fact at each.
Use Chinese for progress, reason, and evidence observations. IDs must be exact.'''


INDEPENDENT_FRAME_RULES = (
    RULES.replace('the current chronological contact sheet and the observations from\nthis conversation only',
                  'only the chronologically ordered images in this request')
    .replace(' Once visibly confirmed, retain success unless later\nevidence shows the original observation was mistaken.', '')
    .replace('what\nchanged since the previous check', 'what changed within the current window')
    .replace('frame IDs previously supplied', 'frame IDs listed in this request')
)


def prepare(video, directory, ffmpeg, ffprobe):
    from PIL import Image, ImageDraw

    data = subprocess.check_output([
        ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=nb_frames,avg_frame_rate,duration', '-of', 'json', str(video),
    ])
    stream, = json.loads(data)['streams']
    fps = Fraction(stream['avg_frame_rate'])
    count = int(stream['nb_frames'])
    if fps != 30 or count < 1:
        raise ValueError(f'{video.name}: expected a nonempty 30 fps cam_high clip')
    frames_per_window = int(fps * WINDOW_SECONDS)
    groups = [sorted(set(range(start, min(start + frames_per_window, count), SAMPLE_EVERY_FRAMES))
                     | {min(start + frames_per_window, count) - 1})
              for start in range(0, count, frames_per_window)]
    numbers = [number for group in groups for number in group]
    frame_dir = directory / 'frames'
    sheet_dir = directory / 'sheets'
    frame_dir.mkdir(parents=True)
    sheet_dir.mkdir()
    expression = '+'.join(f'eq(n,{number})' for number in numbers)
    subprocess.run([
        ffmpeg, '-v', 'error', '-nostdin', '-threads', '2', '-i', str(video), '-map', '0:v:0',
        '-vf', f"select='{expression}'", '-fps_mode', 'vfr', '-q:v', '2', str(frame_dir / '%06d.jpg'),
    ], check=True)
    extracted = sorted(frame_dir.glob('*.jpg'))
    if len(extracted) != len(numbers):
        raise ValueError(f'{video.name}: expected {len(numbers)} samples, got {len(extracted)}')
    frames = {}
    for path, number in zip(extracted, numbers):
        frame_id = f'F{number:06d}'
        destination = frame_dir / f'{frame_id}.jpg'
        path.rename(destination)
        frames[frame_id] = {'path': str(destination), 'time_s': round(float(number / fps), 3)}
    windows = []
    width, height, columns = 384, 312, 4
    for index, group in enumerate(groups):
        ids = [f'F{number:06d}' for number in group]
        rows = (len(ids) + columns - 1) // columns
        sheet = Image.new('RGB', (width * columns, height * rows), '#151719')
        draw = ImageDraw.Draw(sheet)
        for position, frame_id in enumerate(ids):
            frame = frames[frame_id]
            left, top = position % columns * width, position // columns * height
            with Image.open(frame['path']) as original:
                sheet.paste(original.resize((width, 288), Image.Resampling.LANCZOS), (left, top))
            draw.text((left + 8, top + 293), f'{frame_id}   {frame["time_s"]:.3f}s', fill='white')
        path = sheet_dir / f'{index:03d}.jpg'
        sheet.save(path, quality=91)
        windows.append({'index': index, 'sheet': str(path), 'frame_ids': ids,
                        'start_s': frames[ids[0]]['time_s'], 'observed_at_s': frames[ids[-1]]['time_s'],
                        'end_of_episode': index == len(groups) - 1})
    return {'episode': video.stem, 'video': str(video), 'duration_s': float(Fraction(count, 30)),
            'sha256': hashlib.sha256(video.read_bytes()).hexdigest(), 'frames': frames, 'windows': windows}


def reuse_frames(config_path, videos, output):
    source = json.loads(config_path.read_text(encoding='utf-8'))
    if source['window_seconds'] != WINDOW_SECONDS or source['sample_every_frames'] != SAMPLE_EVERY_FRAMES:
        raise ValueError('Saved frame index does not match the current sampling settings')
    episodes_by_id = {episode['episode']: episode for episode in source['episodes']}
    episodes = [episodes_by_id[video.stem] for video in videos]
    for episode, video in zip(episodes, videos):
        if hashlib.sha256(video.read_bytes()).hexdigest() != episode['sha256']:
            raise ValueError(f'{video.name}: does not match the saved frame index')
        episode['video'] = str(video)
        destination = output / video.stem / 'frames'
        destination.mkdir(parents=True)
        for frame_id, frame in episode['frames'].items():
            filename = f'{frame_id}.jpg'
            original = config_path.parent / video.stem / 'frames' / filename
            path = destination / filename
            shutil.copyfile(original, path)
            frame['path'] = str(path)
        for window in episode['windows']:
            window.pop('sheet', None)
    return episodes


class Astra:
    """Use the local Codex app server's structured turns and ephemeral threads."""

    def __init__(self, process, timeout):
        self.process = process
        self.timeout = timeout
        self.pending = {}
        self.notifications = {}
        self.next_id = 0
        self.reader = asyncio.create_task(self.read())

    async def send(self, message):
        self.process.stdin.write((json.dumps(message, ensure_ascii=False) + '\n').encode())
        await self.process.stdin.drain()

    async def request(self, method, params):
        self.next_id += 1
        request_id = self.next_id
        response = asyncio.get_running_loop().create_future()
        self.pending[request_id] = response
        try:
            await self.send({'method': method, 'id': request_id, 'params': params})
            return await asyncio.wait_for(response, self.timeout)
        finally:
            self.pending.pop(request_id, None)

    async def read(self):
        while line := await self.process.stdout.readline():
            message = json.loads(line)
            method, request_id = message.get('method'), message.get('id')
            params = message.get('params', {})
            if request_id is not None and method is None:
                response = self.pending.get(request_id)
                if response is not None and not response.done():
                    if 'error' in message:
                        response.set_exception(RuntimeError(str(message['error'])))
                    else:
                        response.set_result(message['result'])
            elif request_id is not None:
                await self.send({'id': request_id, 'error': {'code': -32601, 'message': 'Tool requests are disabled'}})
            elif method in {'item/agentMessage/delta', 'item/completed', 'turn/completed'} and params.get('threadId') in self.notifications:
                self.notifications[params['threadId']].put_nowait((method, params, time.monotonic()))
        for response in self.pending.values():
            if not response.done():
                response.set_exception(RuntimeError('Codex app server closed its output'))
        for queue in self.notifications.values():
            queue.put_nowait(('closed', {}, time.monotonic()))

    async def start(self, cwd, model=MODEL):
        response = await self.request('thread/start', {
            'model': model, 'cwd': cwd, 'approvalPolicy': 'never', 'sandbox': 'read-only',
            'ephemeral': True, 'developerInstructions': RULES,
        })
        if response['model'] != model:
            raise RuntimeError(f'Requested {model}, app server selected {response["model"]}')
        thread_id = response['thread']['id']
        self.notifications[thread_id] = asyncio.Queue()
        return thread_id

    async def observe(self, thread_id, text, sheet=None):
        payload = {'threadId': thread_id, 'effort': EFFORT, 'input': [{'type': 'text', 'text': text}]}
        if sheet is not None:
            payload['outputSchema'] = OUTPUT_SCHEMA
            payload['input'].append({'type': 'localImage', 'path': sheet})
        submitted = time.monotonic()
        turn = await self.request('turn/start', payload)
        ack_seconds = round(time.monotonic() - submitted, 3)
        turn_id = turn['turn']['id']
        queue = self.notifications[thread_id]
        last_message = None
        first_text_seconds = None
        unexpected = []
        while True:
            method, params, received = await asyncio.wait_for(queue.get(), self.timeout)
            if method == 'closed':
                raise RuntimeError('Codex app server closed before the turn completed')
            if method == 'turn/completed' and params['turn']['id'] == turn_id:
                result = params['turn']
                if result['status'] != 'completed' or not last_message:
                    raise RuntimeError(f'Astra turn did not complete with a response: {result.get("error") or result["status"]}')
                if unexpected:
                    raise RuntimeError(f'Unexpected tool activity: {unexpected}')
                return turn_id, last_message, {
                    'turn_ack_seconds': ack_seconds,
                    'first_text_seconds': first_text_seconds,
                    'turn_completed_seconds': round(received - submitted, 3),
                }
            if (method == 'item/agentMessage/delta' and params['turnId'] == turn_id
                    and first_text_seconds is None and params.get('delta')):
                first_text_seconds = round(received - submitted, 3)
            if method == 'item/completed' and params['turnId'] == turn_id:
                item = params['item']
                if item['type'] == 'agentMessage':
                    last_message = item['text']
                elif item['type'] not in {'userMessage', 'reasoning'}:
                    unexpected.append(item['type'])


def should_stop(backend, row):
    return 'error' in row and not (isinstance(backend, MoTVLM) and row.get('error_kind') == 'invalid_response')


async def observe_window(backend, thread, episode, window, known_ids):
    independent = isinstance(backend, MoTVLM) and backend.independent_frames
    frame_ids = window['frame_ids'][-backend.max_images:] if independent else window['frame_ids']
    known_ids.update(frame_ids)
    frames = [dict(frame_id=frame_id, **episode['frames'][frame_id]) for frame_id in frame_ids]
    text = window_prompt(TASK, window, frames, independent=independent)
    start = time.monotonic()
    row = {**window, 'thread_id': thread, 'request_started_at': datetime.now(timezone.utc).isoformat()}
    if independent:
        row['frame_ids'] = frame_ids
        row['input_images'] = frames
        image_input = [frame['path'] for frame in frames]
    else:
        image_input = window['sheet']
    try:
        turn, raw, timings = await backend.observe(thread, text, image_input)
    except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as error:
        row.update(error=str(error) or type(error).__name__, error_kind='request_error')
    else:
        row.update(turn_id=turn, raw_response=raw, **timings)
        try:
            if row.get('output_limit_reached'):
                raise ValueError('Model response hit the token limit')
            answer = parse_answer(raw)
            validate(answer, frame_ids if independent else known_ids)
            if isinstance(backend, MoTVLM) and not independent:
                backend.remember(thread, answer)
            row['answer'] = {**answer, 'evidence': [dict(item, **episode['frames'][item['frame_id']])
                                                   for item in answer['evidence']]}
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            row.update(error=str(error) or type(error).__name__, error_kind='invalid_response')
    row['request_seconds'] = round(time.monotonic() - start, 3)
    return row


async def warmup_thread(astra, thread):
    turn, answer, measurements = await astra.observe(thread, WARMUP_PROMPT)
    if answer.strip() != 'READY':
        raise RuntimeError(f'Unexpected warmup response for session {thread}: {answer!r}')
    return {'turn_id': turn, 'response': answer, **measurements}


async def start_episode(astra, neutral_cwd, warmup=False):
    started_at = time.monotonic()
    thread = await astra.start(neutral_cwd)
    timing = {'thread_start_seconds': round(time.monotonic() - started_at, 3)}
    if warmup:
        timing['warmup'] = await warmup_thread(astra, thread)
    return thread, started_at, timing


async def evaluate_episode(backend, episode, output, neutral_cwd, session=None):
    name = episode['episode']
    with (output / name / 'timeline.jsonl').open('w', encoding='utf-8') as log:
        thread, started_at, timing = session or await start_episode(backend, neutral_cwd)
        known_ids = set()
        for window in episode['windows']:
            setup_wait = time.monotonic() - started_at
            row = await observe_window(backend, thread, episode, window, known_ids)
            if window['index'] == 0:
                row.update(timing, before_first_video_seconds=round(setup_wait, 3))
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            cutoff = row['observed_at_s']
            state = row['answer']['outcome'] if 'answer' in row else row['error']
            print(f'{name} {cutoff:6.3f}s {state} ({row["request_seconds"]:.1f}s)', flush=True)
            if should_stop(backend, row):
                raise RuntimeError(f'{name} at {cutoff:.3f}s: {row["error"]}')


@asynccontextmanager
async def open_astra(log_path, timeout, default_model=MODEL):
    with tempfile.TemporaryDirectory(prefix='cam-high-vlm-') as cwd, log_path.open('w') as log:
        command = ['codex', 'app-server', '-c', 'web_search="disabled"', '-c', 'mcp_servers={}',
                   '-c', f'model={json.dumps(default_model)}']
        for feature in ('shell_tool', 'apps', 'multi_agent', 'view_image', 'image_generation', 'memories'):
            command += ['--disable', feature]
        process = await asyncio.create_subprocess_exec(*command, cwd=cwd, stdin=asyncio.subprocess.PIPE,
                                                        stdout=asyncio.subprocess.PIPE, stderr=log, limit=2**22)
        astra = Astra(process, timeout)
        try:
            await astra.request('initialize', {'clientInfo': {'name': 'cam_high_video_eval', 'version': '0.1.0'}})
            await astra.send({'method': 'initialized', 'params': {}})
            yield astra, cwd
        finally:
            if process.returncode is None:
                process.terminate()
            await process.wait()
            await astra.reader


async def open_motvlm(url, timeout, directory, *, independent_frames=False):
    rules = INDEPENDENT_FRAME_RULES if independent_frames else RULES
    backend = MoTVLM(url, timeout, rules, OUTPUT_SCHEMA, independent_frames=independent_frames)
    health = await backend.health()
    (directory / 'health.json').write_text(json.dumps(health, indent=2) + '\n', encoding='utf-8')
    return backend


async def run(episodes, output, args):
    if args.model == MOTVLM_MODEL:
        backend = await open_motvlm(args.motvlm_url, args.timeout, output,
                                    independent_frames=args.motvlm_frames_from is not None)
        for item in episodes:
            await evaluate_episode(backend, item, output, None)
        return []
    async with open_astra(output / 'codex.log', args.timeout) as (astra, cwd):
        if args.serial:
            for item in episodes:
                session = await start_episode(astra, cwd, warmup=args.warmup)
                if args.warmup:
                    print(f'{item["episode"]} Astra warmup complete. Sending its first window.', flush=True)
                await evaluate_episode(astra, item, output, cwd, session)
            return []
        sessions = [None] * len(episodes)
        if args.warmup:
            sessions = await asyncio.gather(*(start_episode(astra, cwd, warmup=True) for _ in episodes))
            print('All sessions warmed. Starting the video checks.', flush=True)
        results = await asyncio.gather(*(evaluate_episode(astra, item, output, cwd, session)
                                         for item, session in zip(episodes, sessions)),
                                       return_exceptions=True)
        return [str(result) for result in results if isinstance(result, Exception)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, default=ROOT / 'videos/box_pick_right_cam_high_5_episodes')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', choices=[MODEL, MOTVLM_MODEL], default=MODEL)
    parser.add_argument('--motvlm-url', help='Override the endpoint configured in image_eval/tasks.json.')
    parser.add_argument('--motvlm-frames-from', type=Path, metavar='CONFIG',
                        help='Reuse a saved frame index for an offline MoTVLM run with separate images and no history.')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--serial', action='store_true', help='Run one Astra episode and one model request at a time.')
    parser.add_argument('--warmup', action=argparse.BooleanOptionalAction, default=True,
                        help='Send one text-only warmup in each episode thread before evaluating (default: on).')
    args = parser.parse_args()
    if args.motvlm_frames_from and args.model != MOTVLM_MODEL:
        parser.error('--motvlm-frames-from requires --model motvlm')
    videos = [args.videos.resolve() / f'episode_{index:06d}.mp4' for index in range(5)]
    for video in videos:
        if not video.is_file():
            parser.error(f'Missing episode: {video}')
    if args.model == MOTVLM_MODEL:
        args.warmup = False
        args.motvlm_url = configured_url(args.motvlm_url)
    default_output = ROOT / 'results' / ('video-motvlm-v1' if args.model == MOTVLM_MODEL else 'video-astra-v1')
    if args.motvlm_frames_from:
        default_output = ROOT / 'results' / 'video-motvlm-frames-no-history'
    output = (args.output or default_output).resolve()
    output.mkdir(parents=True)
    if args.motvlm_frames_from:
        episodes = reuse_frames(args.motvlm_frames_from.resolve(), videos, output)
    else:
        episodes = [prepare(video, output / video.stem, args.ffmpeg, args.ffprobe) for video in videos]
    config = {'task': TASK, 'model': args.model, 'effort': EFFORT if args.model == MODEL else None,
              'rules': RULES, 'schema': OUTPUT_SCHEMA,
              'warmup_prompt': WARMUP_PROMPT if args.warmup else None,
              'parallel_episodes': 1 if args.serial or args.model == MOTVLM_MODEL else len(episodes),
              'window_seconds': WINDOW_SECONDS, 'sample_every_frames': SAMPLE_EVERY_FRAMES,
              'episodes': episodes}
    if args.model == MOTVLM_MODEL:
        config.update(motvlm_url=args.motvlm_url, history='recent accepted observations plus first success')
    if args.motvlm_frames_from:
        config.update(rules=INDEPENDENT_FRAME_RULES, history='none', input_mode='independent_frames')
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    try:
        failures = asyncio.run(run(episodes, output, args))
    finally:
        write_report(output, config)
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Report: {output / "report.html"}', flush=True)


if __name__ == '__main__':
    main()
