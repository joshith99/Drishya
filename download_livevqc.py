import requests
import re
from pathlib import Path

s = requests.Session()
s.headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120'
URL = 'https://utexas.box.com/s/2vvm0gm684nggnkmzemb1q6tdqt9pii8'
PWD = 'LiveVQC2019'
OUT = Path('/workspace/datasets/live_vqc')
OUT.mkdir(parents=True, exist_ok=True)

print('[1] Fetching Box page...')
r = s.get(URL)
print(f'    HTTP {r.status_code}')

# Try to extract request_token from hidden form field
tok = re.search(r"name=['\"]request_token['\"] value=['\"]([^'\"]+)", r.text)
if not tok:
    tok = re.search(r'"requestToken"\s*:\s*"([^"]+)"', r.text)
tkval = tok.group(1) if tok else ''
print(f'    token={tkval[:30]}...' if tkval else '    token NOT FOUND')

print('[2] Posting password...')
resp = s.post(URL, data={'password': PWD, 'request_token': tkval},
              headers={'Referer': URL,
                       'Content-Type': 'application/x-www-form-urlencoded'})
print(f'    POST {resp.status_code} -> {resp.url[:80]}')

print('[3] Trying download...')
resp = s.get(URL + '?dl=1', stream=True, allow_redirects=True)
ctype = resp.headers.get('Content-Type', '')
clen = int(resp.headers.get('Content-Length', 0))
print(f'    Content-Type={ctype}')
print(f'    Content-Length={round(clen/1e9, 2)} GB')

if clen > 100_000_000 or 'zip' in ctype or 'octet' in ctype:
    out_f = OUT / 'live_vqc.zip'
    done = 0
    with open(out_f, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                done += len(chunk)
                pct = done / clen * 100 if clen else 0
                print(f'    {pct:.1f}% ({done/1e9:.2f} GB)', flush=True)
    print(f'[DONE] Saved to {out_f}')
else:
    print('[WARN] Response does not look like a zip. Saving debug page...')
    debug = OUT / 'box_debug.html'
    debug.write_text(resp.text[:10000])
    print(f'    Saved to {debug}')
    # Look for any direct download links
    links = re.findall(r'href=["\']([^"\']*download[^"\']*)["\']', resp.text, re.I)
    print(f'    Download links found: {links[:5]}')
    print(resp.text[:500])
