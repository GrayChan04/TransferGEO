"""Conservative JSON serialization repair, without completing missing content."""
import json
import re


def parse_json_format_only(text):
    original = text
    candidate = text.strip()
    actions = []
    if candidate.endswith('<|user|>'):
        candidate = candidate[:-8].rstrip()
        actions.append('remove_terminal_user_token')
    fence = re.fullmatch(r'```json\s*\n(?P<body>.*?)\n```', candidate, re.S)
    if fence:
        candidate = fence['body']
        actions.append('remove_single_json_fence')
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        # Only one string-valued property on a complete line is eligible.
        # Reject apparent extra properties instead of swallowing their syntax.
        pattern = re.compile(r'^(?P<head>\s*"[^"\\]+"\s*:\s*")(?P<body>.*)(?P<end>"\s*,?\s*)$')
        fixed = []
        for line in candidate.splitlines(keepends=True):
            match = pattern.fullmatch(line)
            if match is None:
                fixed.append(line)
                continue
            try:
                json.loads('{'+line.strip().removesuffix(',')+'}')
            except json.JSONDecodeError:
                pass
            else:
                fixed.append(line)
                continue
            body = match['body']
            if re.search(r'"\s*[,}:]|"\s*:', body):
                raise ValueError('Ambiguous quote boundary: refusing repair')
            output = []
            slashes = 0
            count = 0
            for char in body:
                if char == '"' and slashes % 2 == 0:
                    output.append('\\')
                    count += 1
                output.append(char)
                slashes = slashes + 1 if char == '\\' else 0
            if count:
                actions.append({'escape_internal_double_quotes':count})
            fixed.append(match['head']+''.join(output)+match['end'])
        candidate = ''.join(fixed)
        # No closing braces, commas, fields, or values are ever synthesized.
        payload = json.loads(candidate)
    return payload, {'policy':'json_format_only_v1', 'before':original,
                     'after':candidate, 'actions':actions,
                     'content_completion':False}
