"""Schema and interpretation for video-monitoring responses."""

import json
import re

PHASES = ['waiting', 'approaching', 'grasping', 'lifting', 'retrying', 'after_pick', 'unclear']
OUTCOMES = ['in_progress', 'success', 'failure', 'uncertain']
ACTIONS = ['continue', 'retry', 'intervene', 'finish', 'observe']


def object_schema(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


STRING = {'type': 'string'}
OUTPUT_SCHEMA = object_schema({
    'phase': {'type': 'string', 'enum': PHASES},
    'progress': STRING,
    'outcome': {'type': 'string', 'enum': OUTCOMES},
    'needs_intervention': {'type': 'boolean'},
    'decision': {'type': 'string', 'enum': ACTIONS},
    'reason': STRING,
    'evidence': {'type': 'array', 'items': object_schema({'frame_id': STRING, 'observation': STRING})},
})


def validate(answer, known_ids):
    if not isinstance(answer, dict) or set(answer) != set(OUTPUT_SCHEMA['required']):
        raise ValueError('Model response does not match the required output fields')
    if answer['phase'] not in PHASES or answer['outcome'] not in OUTCOMES or answer['decision'] not in ACTIONS:
        raise ValueError('Unknown model decision, outcome or phase')
    if type(answer['needs_intervention']) is not bool:
        raise ValueError('needs_intervention must be boolean')
    evidence = answer['evidence']
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 2 or any(
        not isinstance(item, dict) or not isinstance(item.get('frame_id'), str) or item['frame_id'] not in known_ids
        or not isinstance(item.get('observation'), str) or not item['observation'].strip() for item in evidence
    ):
        raise ValueError('Model did not cite one or two valid observed frames')
    if answer['needs_intervention'] != (answer['decision'] == 'intervene'):
        raise ValueError('Model intervention judgment and decision disagree')


def parse_answer(raw, *, object_pairs_hook=None):
    if not isinstance(raw, str):
        raise ValueError('Model returned no text')
    text = raw.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, flags=re.IGNORECASE | re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text, object_pairs_hook=object_pairs_hook)


def resolve_frame(reference, frames):
    if type(reference) not in (str, int, float):
        return None
    value = str(reference).strip()
    if value in frames:
        return frames[value]
    if re.fullmatch(r'[0-9]{1,6}', value):
        return frames.get(f'F{int(value):06d}')
    if re.fullmatch(r'[0-9]+\.[0-9]+s?|[0-9]+s', value):
        timestamp = float(value.removesuffix('s'))
        matches = [frame for frame in frames.values() if frame['time_s'] == timestamp]
        if len(matches) == 1:
            return matches[0]
    return None


def interpret_independent_response(raw, input_images):
    """Recover structured fields using only the images sent in that request."""
    response = parse_answer(raw)
    if not isinstance(response, dict):
        raise ValueError('Expected a JSON object')
    answer = {}
    for name in ('phase', 'outcome', 'decision'):
        value = response.get(name)
        choices = OUTPUT_SCHEMA['properties'][name]['enum']
        answer[name] = value if value in choices else None
    for name in ('progress', 'reason'):
        value = response.get(name)
        answer[name] = value if isinstance(value, str) and value.strip() else None
    intervention = response.get('needs_intervention')
    answer['needs_intervention'] = intervention if type(intervention) is bool else None

    frames = {image['frame_id']: image for image in input_images}
    evidence = response.get('evidence')
    resolved = {}
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            frame = resolve_frame(item.get('frame_id'), frames)
            observation = item.get('observation')
            if frame and isinstance(observation, str) and observation.strip():
                resolved.setdefault(frame['frame_id'], {**frame, 'observation': observation})
    if all(value is None for value in answer.values()) and not resolved:
        raise ValueError('Response contained no recognizable judgment')
    answer['evidence'] = list(resolved.values())

    missing = [name for name, value in answer.items() if value is None]
    if not resolved:
        missing.append('evidence')
    conflicts = []
    if answer['decision'] == 'finish' and answer['outcome'] not in (None, 'success'):
        conflicts.append('outcome_decision')
    if (answer['decision'] is not None and answer['needs_intervention'] is not None
            and answer['needs_intervention'] != (answer['decision'] == 'intervene')):
        conflicts.append('intervention_decision')
    return answer, missing, conflicts
