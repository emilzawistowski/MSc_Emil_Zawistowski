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
    print(f"\n{'='*50}")
    print(f"Block {block}: {meta['block_name']}")
    print(f"  Total number of trials: {len(trials)}")

    # 1. Check trigger codes
    codes = Counter(str(t['trigger_code']) for t in trials)
    print(f"  Trigger codes: {dict(codes)}")

    expected_codes = EXPECTED.get(block, {})
    for code, meaning in expected_codes.items():
        if code not in codes:
            print(f"  ERROR: Missing code {code} ({meaning})!")
        else:
            print(f"  OK: Code {code} ({meaning}): {codes[code]} times")

    # 2. Check if onset_sample is monotonically increasing
    onsets = [t['onset_sample'] for t in trials]
    if onsets != sorted(onsets):
        print("  ERROR: onset_sample is not monotonically increasing!")
    else:
        print(f"  OK: Onset samples OK (first: {onsets[0]}, last: {onsets[-1]})")

    # 3. Check SOA
    soa_samples = set()
    for i in range(1, min(20, len(trials))):
        diff = trials[i]['onset_sample'] - trials[i-1]['onset_sample']
        soa_samples.add(diff)
    if len(soa_samples) == 1:
        soa_ms = list(soa_samples)[0] / meta['fs'] * 1000
        print(f"  OK: Constant SOA: {soa_ms:.1f} ms")
    else:
        print(f"  ERROR: Non-uniform SOA: {soa_samples}")

    # 4. Check lead-in
    lead_ins = [t for t in trials if t.get('is_lead_in')]
    print(f"  OK: Lead-in trials: {len(lead_ins)}")

    # 5. Check that deviants are not consecutive
    if block in ['A', 'B']:
        dev_code = 12 if block == 'A' else 22
        consecutive = 0
        for i in range(1, len(trials)):
            if (trials[i]['trigger_code'] == dev_code and
                trials[i-1]['trigger_code'] == dev_code):
                consecutive += 1
        if consecutive > 0:
            print(f"  ERROR: {consecutive} pairs of consecutive deviants!")
        else:
            print(f"  OK: No consecutive deviants")

    # 6. Block C: check distance distribution
    if block == 'C':
        dists = Counter(str(round(t['distance_m'], 1)) for t in trials)
        print(f"  Distance distribution: {dict(dists)}")

    return True

if __name__ == '__main__':
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./rendered_blocks')
    json_files = sorted(folder.glob('*_metadata.json'))
    if not json_files:
        print(f"No JSON files in {folder}")
        sys.exit(1)
    for jf in json_files:
        check_block(jf)
    print("\nOK: Selfcheck completed")