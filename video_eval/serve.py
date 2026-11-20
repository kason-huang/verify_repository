#!/usr/bin/env python3
"""Serve the live monitoring page on localhost while Astra is evaluating."""

import argparse
import hashlib
import json
import math
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, unquote, urlsplit

from live import LiveRun
from reporting import VIEWER, report_data
from run import MODEL, TASK

ROOT = Path(__file__).resolve().parents[1]
LIVE_VIEWER = Path(__file__).with_name('live.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/video-astra-v1')
    parser.add_argument('--videos', type=Path, default=ROOT / 'videos/box_pick_right_cam_high_5_episodes')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model', action='append', default=[], help='Enable an additional Codex model ID in the live viewer.')
    args = parser.parse_args()
    output, videos = args.output.resolve(), args.videos.resolve()
    models = list(dict.fromkeys([MODEL, *args.model]))
    active = None
    lock = Lock()

    def load_config():
        return json.loads((output / 'config.json').read_text(encoding='utf-8'))

    def load_episode(episode_id):
        episode = next(item for item in load_config()['episodes'] if item['episode'] == episode_id)
        local_video = videos / f'{episode["episode"]}.mp4'
        if hashlib.sha256(local_video.read_bytes()).hexdigest() != episode['sha256']:
            raise ValueError('Video has changed')
        if any(not Path(window['sheet']).is_file() for window in episode['windows']):
            raise ValueError('Missing frame sheets')
        return episode

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            target = urlsplit(self.path)
            url = unquote(target.path)
            if url == '/':
                self.reply(VIEWER.read_bytes(), 'text/html; charset=utf-8')
            elif url == '/live':
                self.reply(LIVE_VIEWER.read_bytes(), 'text/html; charset=utf-8')
            elif url == '/live/options':
                try:
                    config = load_config()
                except (FileNotFoundError, json.JSONDecodeError):
                    self.json_reply({'error': '找不到评测帧索引，请先准备这 5 段视频。'}, 503)
                    return
                self.json_reply({'task': TASK, 'models': models, 'window_seconds': config['window_seconds'],
                                 'episodes': [{'id': item['episode'], 'duration_s': item['duration_s'],
                                               'video': f'/video/{item["episode"]}.mp4'} for item in config['episodes']]})
            elif url == '/live/state':
                run_id = parse_qs(target.query).get('id', [''])[0]
                with lock:
                    current = active
                if current is None or current.id != run_id:
                    self.json_reply({'error': '找不到这个会话。'}, 404)
                    return
                self.json_reply(current.snapshot())
            elif url == '/live/active':
                with lock:
                    current = active
                state = current.snapshot() if current is not None else None
                self.json_reply(state if state and state['status'] not in {'complete', 'error', 'stopped'} else None)
            elif url == '/api':
                try:
                    data = report_data(output, load_config(), live=True)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = {'episodes': []}
                self.json_reply(data)
            elif url.startswith('/video/'):
                self.file(videos, url.removeprefix('/video/'), r'episode_\d{6}\.mp4')
            elif url.startswith('/media/'):
                self.file(output, url.removeprefix('/media/'), r'episode_\d{6}/(?:frames/F\d{6}|sheets/\d{3})\.jpg')
            else:
                self.send_error(404)

        def do_POST(self):
            nonlocal active
            host = self.headers.get('Host')
            port = self.server.server_port
            if host not in {f'127.0.0.1:{port}', f'localhost:{port}'} or self.headers.get('Origin') != f'http://{host}':
                self.json_reply({'error': '请从本地试跑页面操作。'}, 403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096 or self.headers.get_content_type() != 'application/json':
                    raise ValueError('Invalid JSON request')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Expected an object')
            except (ValueError, UnicodeDecodeError):
                self.json_reply({'error': '无效的请求。'}, 400)
                return
            if self.path == '/live/start':
                model = body.get('model')
                warmup = body.get('warmup', False)
                if model not in models:
                    self.json_reply({'error': '此模型未配置。'}, 400)
                    return
                if type(warmup) is not bool:
                    self.json_reply({'error': '无效的预热选项。'}, 400)
                    return
                try:
                    episode_id = body.get('episode')
                    episode = None if warmup and episode_id is None else load_episode(episode_id)
                except (KeyError, StopIteration, OSError, ValueError):
                    self.json_reply({'error': '视频和抽帧索引不匹配，请重新准备评测输入。'}, 400)
                    return
                with lock:
                    if active is not None and active.worker.is_alive():
                        state = active.snapshot()
                        if (warmup and episode is None and state['episode'] is None and state['model'] == model
                                and state['status'] in {'connecting', 'warming', 'ready'}):
                            self.json_reply(state)
                            return
                        self.json_reply({'error': '已有会话在运行。请先停止并等待当前请求结束。'}, 409)
                        return
                    active = current = LiveRun(episode, model, output, warmup=warmup)
            elif self.path in {'/live/play', '/live/advance', '/live/stop'}:
                with lock:
                    current = active
                if current is None or current.id != body.get('id'):
                    self.json_reply({'error': '找不到这个会话。'}, 404)
                    return
                if self.path == '/live/play':
                    try:
                        episode = load_episode(body.get('episode'))
                    except (KeyError, StopIteration, OSError, ValueError):
                        self.json_reply({'error': '视频和抽帧索引不匹配，请重新准备评测输入。'}, 400)
                        return
                    try:
                        current.select_episode(episode)
                    except ValueError as error:
                        self.json_reply({'error': str(error)}, 409)
                        return
                elif self.path == '/live/advance':
                    position, ended = body.get('playhead_s'), body.get('ended', False)
                    if current.episode is None:
                        self.json_reply({'error': '请先选择视频并开始监控。'}, 409)
                        return
                    duration = current.episode['duration_s']
                    if (type(position) not in (float, int) or not 0 <= position <= duration + 0.1
                            or not math.isfinite(position) or type(ended) is not bool
                            or (ended and position < duration - 0.1)):
                        self.json_reply({'error': '无效的播放位置。'}, 400)
                        return
                    try:
                        current.advance(position, ended)
                    except ValueError as error:
                        self.json_reply({'error': str(error)}, 409)
                        return
                else:
                    current.stop()
            else:
                self.send_error(404)
                return
            self.json_reply(current.snapshot())

        def json_reply(self, data, status=200):
            self.reply(json.dumps(data, ensure_ascii=False).encode(), 'application/json; charset=utf-8', status=status)

        def file(self, root, name, pattern):
            path = (root / name).resolve()
            if not re.fullmatch(pattern, name) or not path.is_relative_to(root) or not path.is_file():
                self.send_error(404)
                return
            media_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'
            if not (header := self.headers.get('Range')):
                self.reply(path.read_bytes(), media_type)
                return
            size = path.stat().st_size
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            first, last = match.groups()
            start = int(first) if first else max(size - int(last), 0)
            end = min(int(last), size - 1) if first and last else size - 1
            if not 0 <= start <= end < size:
                self.send_error(416)
                return
            with path.open('rb') as stream:
                stream.seek(start)
                self.reply(stream.read(end - start + 1), media_type, f'bytes {start}-{end}/{size}')

        def reply(self, data, content_type, content_range=None, status=200):
            self.send_response(206 if content_range else status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Accept-Ranges', 'bytes')
            if content_range:
                self.send_header('Content-Range', content_range)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f'http://127.0.0.1:{server.server_port}/', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
