"""Build an allowlisted, static-only website; never copy data or credentials."""
import argparse
import shutil
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / 'backend' / 'web'
ASSETS = ('app.js', 'style.css', 'refinements.css', 'ble_frames.js', 'ble_signals.js', 'waveforms.js')

def build(destination):
    destination = Path(destination)
    (destination / 'static').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WEB / 'index.html', destination / 'index.html')
    for name in ASSETS:
        shutil.copyfile(WEB / name, destination / 'static' / name)
    (destination / '.nojekyll').write_text('', encoding='utf-8')
    return destination

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='site-build')
    build(parser.parse_args().output)
