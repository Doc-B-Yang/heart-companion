"""Read the user-supplied PPTX and proposal without modifying them."""
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent / 'diagnostics' / 'acceptance-2026-10-03'
out.mkdir(parents=True, exist_ok=True)
ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
deck = root / '基于实时心率变异性驱动的主动式情感陪伴系统(2).pptx'
proposal = root / '基于实时心率变异性驱动的主动式情感陪伴系统——大创项目立项报告(1).docx'
sources = {'pptx': str(deck), 'docx': str(proposal)}
slides = []
with zipfile.ZipFile(deck) as z:
    names = [n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]
    for name in sorted(names, key=lambda s: int(re.search(r'(\d+)\.xml', s).group(1))):
        xml = ET.fromstring(z.read(name))
        paras = [''.join(p.itertext()) for p in []]
        paras = [''.join(t.text or '' for t in p.findall('.//a:t', ns))
                 for p in xml.findall('.//a:p', ns)]
        slides.append({'slide': int(re.search(r'(\d+)\.xml', name).group(1)),
                       'paragraphs': [p for p in paras if p],
                       'pictures': len(xml.findall('.//{http://schemas.openxmlformats.org/presentationml/2006/main}pic'))})
with zipfile.ZipFile(proposal) as z:
    xml = ET.fromstring(z.read('word/document.xml'))
    paras = [''.join(t.text or '' for t in p.findall('.//w:t', ns))
             for p in xml.findall('.//w:p', ns)]
    paras = [p for p in paras if p]
result = {'sources': sources, 'slides': slides, 'proposal_paragraphs': paras}
(out / 'source-extract.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
txt = '\n\n'.join(['SLIDE ' + str(s['slide']) + '\n' + '\n'.join(s['paragraphs']) for s in slides])
txt += '\n\nPROPOSAL\n' + '\n'.join('P' + str(i+1) + ' ' + p for i, p in enumerate(paras))
(out / 'source-extract.txt').write_text(txt, encoding='utf-8')
print('SLIDES', len(slides), 'DOC_PARAGRAPHS', len(paras), 'OUTPUT', out)
