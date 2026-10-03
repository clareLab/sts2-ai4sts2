import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / 'src/ai4sts2.json').read_text())
project = ET.parse(root / 'src/ai4sts2.csproj')
props = ET.parse(root / 'Directory.Build.props')
assert manifest['id'] == 'ai4sts2' and manifest['name'] == 'AI4STS2'
assert manifest['version'] == project.findtext('.//Version')
assert manifest['author'] == props.findtext('.//Authors') == 'clareLab'
assert manifest['dependencies'] == []
assert manifest['has_dll'] and not manifest['has_pck'] and manifest['affects_gameplay']
assert all(reference.findtext('Private') == 'false' for reference in project.findall('.//Reference'))
if '--source' not in sys.argv:
    directory = root / 'artifacts/dist'
    expected = {'ai4sts2/ai4sts2.dll', 'ai4sts2/ai4sts2.json', 'ai4sts2/LICENSE'}
    with ZipFile(directory / f"ai4sts2-{manifest['version']}.zip") as package:
        assert set(package.namelist()) == expected
        assert json.loads(package.read('ai4sts2/ai4sts2.json')) == manifest
        assert package.read('ai4sts2/ai4sts2.dll') == (directory / 'ai4sts2/ai4sts2.dll').read_bytes()
        assert package.read('ai4sts2/LICENSE') == (root / 'LICENSE').read_bytes()
print('PASS project and package checks')
