"""Apply explicit, hash-bound source-prefix truncations; never truncate prompts."""
import hashlib
import json


def input_hash(item, documents):
    value = dict(benchmark=item['benchmark'], domain=item['domain'],
                 query=item['query'], documents=documents)
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def apply_source_truncations(conditions, path, protocol):
    if path is None:
        return conditions, {}
    with path.open() as handle:
        rows=[json.loads(l) for l in handle]
    overrides={(r['sample_id'],r['prompt_id']):r for r in rows}
    if len(overrides)!=len(rows):raise ValueError('Duplicate truncation keys')
    used=set(); result=[]
    for item,prompt,docs in conditions:
        key=(item['sample_id'],prompt)
        if key in overrides:
            row=overrides[key]
            if row['protocol']!=protocol or row['original_input_sha256']!=input_hash(item,docs):
                raise ValueError('Truncation input/protocol mismatch')
            shortened=row['documents']
            if len(shortened)!=len(docs) or any(not t or not s.startswith(t) for s,t in zip(docs,shortened)):
                raise ValueError('Truncation must preserve every source as a nonempty prefix')
            if row['truncated_input_sha256']!=input_hash(item,shortened):
                raise ValueError('Truncation output hash mismatch')
            docs=shortened;used.add(key)
        result.append((item,prompt,docs))
    if used!=set(overrides):raise ValueError('Unused truncation entries')
    return result,dict(source_truncation_count=len(used),source_truncation_path=str(path),
        source_truncation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_truncation_policy='shared_three_models_proportional_source_prefix_v1')
