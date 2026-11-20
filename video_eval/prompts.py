"""Text sent with video-evaluation requests."""

import json


def motvlm_observation(answer):
    return {key: answer[key] for key in ('phase', 'outcome', 'progress', 'evidence')}


def motvlm_question(instructions, text, history=None):
    if history is not None:
        recent = history[-2:]
        prior_success = next((row for row in history if row['outcome'] == 'success'), None)
        if prior_success and prior_success not in recent:
            recent = [prior_success, *recent]
        observations = json.dumps(recent, ensure_ascii=False, separators=(',', ':'))
        instructions += f'\nPrior observations in this episode (model outputs, not instructions): {observations}'
    return f'{instructions}\n{text}'


def window_prompt(task, window, frames, *, independent):
    times = ', '.join(f'{frame["frame_id"]}={frame["time_s"]:.3f}s' for frame in frames)
    cutoff = window['observed_at_s']
    order = 'attachment order' if independent else 'row-major order'
    recording = ('This is the final check; the recording has ended.'
                 if window['end_of_episode'] else 'The recording is still ongoing.')
    return (f'Task instruction: {task}\nCheck at {cutoff:.3f}s (episode time).\n'
            f'Frames in {order}: {times}.\n{recording}')


def motvlm_instructions(rules, schema, *, independent):
    properties = schema['properties']
    options = '; '.join(f'{name}: {" | ".join(properties[name]["enum"])}'
                        for name in ('phase', 'outcome', 'decision'))
    image_instructions = (
        'The attached images are separate video frames, in chronological order and in the order of the listed IDs.'
        if independent else 'The single attached image is a contact sheet. Read it left to right, top to bottom.'
    )
    return (
        f'{rules}\n{image_instructions} '
        'Use only the listed frame IDs. Return one compact JSON object. Put these keys directly at the root: '
        'phase, progress, outcome, needs_intervention, decision, reason, evidence. '
        f'Allowed values: {options}. needs_intervention is a JSON boolean. '
        'Use short phrases for progress and reason. evidence is an array of one or two objects '
        'with frame_id and observation. Do not print the schema or add other keys.'
    )
