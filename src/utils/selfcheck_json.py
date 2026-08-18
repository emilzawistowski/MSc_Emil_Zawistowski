"""
Verify JSON metadata files
Run after rendering: python src/utils/selfcheck_json.py ./rendered_blocks
"""
import json, sys
from pathlib import Path
from collections import Counter

EXPECTED = {
    'A': {'11': 'standard', '12': 'deviant'},
    'B': {'21': 'standard', '22': 'deviant'},
    'C': {'31': 'control'},
}

def check_block(json_path):
    with open(json_path) as f:
        meta = json.load(f)

    block = meta['block_type']
    trials = meta['trials']


    codes = Counter(str(t['trigger_code']) for t in trials)


    expected_codes = EXPECTED.get(block, {})
    for code, meaning in expected_codes.items():
        if code not in codes:

            pass
        else:


            pass
    onsets = [t['onset_sample'] for t in trials]
    if onsets != sorted(onsets):

        pass
    else:


        pass
    soa_samples = set()
    for i in range(1, min(20, len(trials))):
        diff = trials[i]['onset_sample'] - trials[i-1]['onset_sample']
        soa_samples.add(diff)
    if len(soa_samples) == 1:
        soa_ms = list(soa_samples)[0] / meta['fs'] * 1000

    else:


        pass
    lead_ins = [t for t in trials if t.get('is_lead_in')]


    if block in ['A', 'B']:
        dev_code = 12 if block == 'A' else 22
        consecutive = 0
        for i in range(1, len(trials)):
            if (trials[i]['trigger_code'] == dev_code and
                trials[i-1]['trigger_code'] == dev_code):
                consecutive += 1
        if consecutive > 0:

            pass
        else:


            pass
    if block == 'C':
        dists = Counter(str(round(t['distance_m'], 1)) for t in trials)


    return True

if __name__ == '__main__':
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./rendered_blocks')
    json_files = sorted(folder.glob('*_metadata.json'))
    if not json_files:

        sys.exit(1)
    for jf in json_files:
        check_block(jf)
