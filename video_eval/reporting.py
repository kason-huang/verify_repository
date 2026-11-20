"""Build the shared data for the live viewer and the offline replay."""

import json
from os.path import relpath
from pathlib import Path
from urllib.parse import quote

VIEWER = Path(__file__).with_name('viewer.html')


def read_rows(path):
    if not path.exists():
        return []
    # A live request can race with a writer at the end of the file.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines(keepends=True)
            if line.endswith('\n')]


def report_data(output, config, live=False):
    episodes = []
    for item in config['episodes']:
        name = item['episode']
        rows = read_rows(output / name / 'timeline.jsonl')
        valid = [row for row in rows if 'answer' in row]
        summary = {
            'episode': name, 'checks_completed': len(valid), 'checks_expected': len(item['windows']),
            'final_outcome': valid[-1]['answer']['outcome'] if valid and valid[-1]['end_of_episode'] else None,
            'first_success_report_s': next((row['observed_at_s'] for row in valid
                                            if row['answer']['outcome'] == 'success'), None),
            'first_intervention_report_s': next((row['observed_at_s'] for row in valid
                                                 if row['answer']['needs_intervention']), None),
            'errors': [row['error'] for row in rows if 'error' in row],
        }
        video = '/video/' + quote(Path(item['video']).name) if live else quote(relpath(item['video'], output))
        for row in rows:
            row['sheet'] = asset_url(output, row['sheet'], live)
            for evidence in row.get('answer', {}).get('evidence', []):
                evidence['path'] = asset_url(output, evidence['path'], live)
        episodes.append({**summary, 'video': video, 'duration_s': item['duration_s'], 'timeline': rows})
    return {'task': config['task'], 'model': config['model'], 'window_seconds': config['window_seconds'],
            'episodes': episodes}


def asset_url(output, path, live):
    return ('/media/' if live else '') + quote(relpath(path, output))


def write_report(output, config):
    data = report_data(output, config)
    embedded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    viewer = VIEWER.read_text(encoding='utf-8').replace('<!-- REPORT_DATA -->', embedded)
    (output / 'report.html').write_text(viewer, encoding='utf-8')
    summary = [{key: value for key, value in row.items() if key not in {'video', 'duration_s', 'timeline'}}
               for row in data['episodes']]
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
