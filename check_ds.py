from datasets import load_dataset
try:
    ds = load_dataset('open-thoughts/OpenThoughts-114k', split='train', streaming=True)
    for i, row in enumerate(ds):
        if i >= 3:
            break
        convs = row.get('conversations', [])
        print(f'--- Record {i} ---')
        for j, c in enumerate(convs):
            from_f = c.get('from', 'MISSING')
            val = c.get('value', '')
            has_think_close = '</think>' in val
            print(f'  turn {j}: from={from_f!r}, len={len(val)}, has_</think>={has_think_close}')
            print(f'    first 200 chars: {repr(val[:200])}')
        print()
except Exception as e:
    print(f'Error: {e}')
