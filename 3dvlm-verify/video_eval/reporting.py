"""Build the shared data for the live viewer and the offline replay."""

import json
from os.path import relpath
from pathlib import Path
from urllib.parse import quote

from motvlm import CONTACT_SHEET_LAYOUT, GENERATE_PATH, MAX_NEW_TOKENS, MODEL_ID
from prompts import motvlm_instructions, motvlm_observation, motvlm_question, window_prompt
from responses import interpret_independent_response, parse_answer

VIEWER = Path(__file__).with_name('viewer.html')


def read_rows(path):
    if not path.exists():
        return []
    # A live request can race with a writer at the end of the file.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines(keepends=True)
            if line.endswith('\n')]


def review_independent_rows(rows):
    for row in rows:
        if not row.get('input_images') or row.get('output_limit_reached') or row.get('error_kind') == 'request_error':
            continue
        try:
            answer, missing, conflicts = interpret_independent_response(row.get('raw_response'), row['input_images'])
        except ValueError:
            continue
        if conflicts or missing or answer != row.get('answer'):
            status = 'conflict' if conflicts else 'partial' if missing else 'normalized'
            row['response_review'] = {
                'status': status, 'missing': missing, 'conflicts': conflicts,
                'original_error': row.get('error'),
            }
        row['answer'] = answer
        row.pop('error', None)
        row.pop('error_kind', None)


def contact_sheet_error(raw, required_fields, observed_ids):
    duplicate_fields = set()

    def record_duplicates(pairs):
        fields = dict(pairs)
        if len(fields) < len(pairs):
            seen = set()
            for key, _ in pairs:
                if key in seen:
                    duplicate_fields.add(key)
                seen.add(key)
        return fields

    try:
        value = parse_answer(raw, object_pairs_hook=record_duplicates)
    except ValueError:
        return ['原始返回无法解析成 JSON。']
    if not isinstance(value, dict):
        return ['原始返回的根节点应为 JSON 对象。']

    reasons = []
    if duplicate_fields:
        reasons.append(f'原始 JSON 重复声明字段 {"、".join(sorted(duplicate_fields))}。')
    missing = set(required_fields) - value.keys()
    if missing:
        reasons.append(f'缺少必填字段 {"、".join(sorted(missing))}。')
    extra = value.keys() - set(required_fields)
    if extra:
        reasons.append(f'包含未定义的根级字段 {"、".join(sorted(extra))}。')
    evidence = value.get('evidence')
    if 'evidence' not in value:
        return reasons
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 2:
        reasons.append('evidence 必须包含一到两条证据。')
        return reasons
    for item in evidence:
        if not isinstance(item, dict):
            reasons.append('evidence 内的每条证据必须是含 frame_id 和 observation 的对象。')
            break
        frame_id = item.get('frame_id')
        if not isinstance(frame_id, str) or frame_id not in observed_ids:
            reasons.append(f'无效的证据帧号 {json.dumps(frame_id, ensure_ascii=False)}，需原样引用已观察的帧号。')
        if not isinstance(item.get('observation'), str) or not item['observation'].strip():
            reasons.append('证据中的 observation 必须是非空字符串。')
    return reasons


def report_data(output, config, live=False, media_prefix='/media/'):
    episodes = []
    independent = config.get('input_mode') == 'independent_frames'
    is_motvlm = config['model'] == MODEL_ID
    instructions = motvlm_instructions(config['rules'], config['schema'], independent=independent) if is_motvlm else None
    for item in config['episodes']:
        name = item['episode']
        rows = read_rows(output / name / 'timeline.jsonl')
        strict_count = sum('answer' in row for row in rows)
        if independent:
            review_independent_rows(rows)
        valid = [row for row in rows if 'answer' in row
                 and row.get('response_review', {}).get('status') not in ('partial', 'conflict')]
        summary = {
            'episode': name, 'checks_completed': len(valid), 'checks_expected': len(item['windows']),
            'final_outcome': valid[-1]['answer']['outcome'] if valid and valid[-1]['end_of_episode'] else None,
            'first_success_report_s': next((row['observed_at_s'] for row in valid
                                            if row['answer']['outcome'] == 'success'), None),
            'first_intervention_report_s': next((row['observed_at_s'] for row in valid
                                                 if row['answer']['needs_intervention']), None),
            'errors': [row['error'] for row in rows if 'error' in row],
        }
        if independent:
            summary['strict_checks_completed'] = strict_count
        video = '/video/' + quote(Path(item['video']).name) if live else quote(relpath(item['video'], output))
        history = []
        observed_ids = set()
        for row in rows:
            observed_ids.update(row['frame_ids'])
            if 'request' not in row and (row.get('input_images') or row.get('sheet')):
                frames = row['input_images'] if independent else [
                    dict(frame_id=frame_id, **item['frames'][frame_id]) for frame_id in row['frame_ids']
                ]
                text = window_prompt(config['task'], row, frames, independent=independent)
                if is_motvlm:
                    row['request'] = {
                        'method': 'POST', 'path': GENERATE_PATH,
                        'question': motvlm_question(instructions, text, None if independent else history),
                        'max_new_tokens': MAX_NEW_TOKENS, 'reconstructed': True,
                    }
                    if not independent:
                        row['request']['layout'] = CONTACT_SHEET_LAYOUT
                else:
                    row['request'] = {'method': 'turn/start', 'text': text, 'effort': config.get('effort'),
                                      'reconstructed': True}
            if is_motvlm and not independent:
                if row.get('error_kind') == 'invalid_response' and not row.get('output_limit_reached'):
                    row['validation_reasons'] = contact_sheet_error(
                        row.get('raw_response'), config['schema']['required'], observed_ids)
                if 'answer' in row:
                    history.append(motvlm_observation(parse_answer(row['raw_response'])))
            if 'sheet' in row:
                row['sheet'] = asset_url(output, row['sheet'], live, media_prefix)
            for image in row.get('input_images', []):
                image['path'] = asset_url(output, image['path'], live, media_prefix)
            for evidence in row.get('answer', {}).get('evidence', []):
                evidence['path'] = asset_url(output, evidence['path'], live, media_prefix)
        episodes.append({**summary, 'video': video, 'duration_s': item['duration_s'], 'timeline': rows})
    return {'task': config['task'], 'model': config['model'], 'window_seconds': config['window_seconds'],
            'episodes': episodes, 'input_mode': config.get('input_mode', 'contact_sheet'),
            'astra_instructions': None if is_motvlm else {
                'rules': config.get('rules'), 'warmup': config.get('warmup_prompt'), 'schema': config.get('schema'),
            }}


def asset_url(output, path, live, media_prefix='/media/'):
    return (media_prefix if live else '') + quote(relpath(Path(path).resolve(), output))


def write_report(output, config):
    data = report_data(output, config)
    embedded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    viewer = VIEWER.read_text(encoding='utf-8').replace('<!-- REPORT_DATA -->', embedded, 1)
    (output / 'report.html').write_text(viewer, encoding='utf-8')
    summary = [{key: value for key, value in row.items() if key not in {'video', 'duration_s', 'timeline'}}
               for row in data['episodes']]
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
