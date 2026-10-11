"""Build an allowlisted, static-only website; never copy data or credentials."""
import argparse
import hashlib
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / 'backend' / 'web'
ASSETS = ('app.js', 'style.css', 'refinements.css', 'ble_frames.js', 'ble_signals.js', 'waveforms.js', 'speech_input.js', 'affect_view.js')

def build(destination):
    destination = Path(destination)
    (destination / 'static').mkdir(parents=True, exist_ok=True)
    sources = {name: (WEB / name).read_text(encoding='utf-8') for name in ASSETS}
    # Version the complete module graph, including imports cached separately by Chrome.
    digest = hashlib.sha256()
    for name, content in sources.items():
        digest.update(name.encode('utf-8') + b'\0' + content.encode('utf-8') + b'\0')
    version = digest.hexdigest()[:16]
    filenames = {name: f'{Path(name).stem}.{version}{Path(name).suffix}' for name in ASSETS}
    index = (WEB / 'index.html').read_text(encoding='utf-8')
    for name, filename in filenames.items():
        index = index.replace(f'./static/{name}', f'./static/{filename}')
    (destination / 'index.html').write_text(index, encoding='utf-8')
    for name in ASSETS:
        content = sources[name]
        if name.endswith('.js'):
            for dependency, filename in filenames.items():
                content = content.replace(f"'./{dependency}'", f"'./{filename}'")
                content = content.replace(f'"./{dependency}"', f'"./{filename}"')
        (destination / 'static' / filenames[name]).write_text(content, encoding='utf-8')
    (destination / '.nojekyll').write_text('', encoding='utf-8')
    return destination

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='site-build')
    build(parser.parse_args().output)
