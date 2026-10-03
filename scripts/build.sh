#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
ai4sts2_game_paths "${1:-}"
ai4sts2_dotnet build src/ai4sts2.csproj -c Release "-p:Sts2DataDir=$data_dir"
python3 - <<'PY'
import hashlib, json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
package = Path('artifacts/dist') / f"ai4sts2-{json.loads(Path('src/ai4sts2.json').read_text())['version']}.zip"
with ZipFile(package, 'w', ZIP_DEFLATED) as archive:
    for file in [Path('artifacts/dist/ai4sts2/ai4sts2.dll'), Path('artifacts/dist/ai4sts2/ai4sts2.json'), Path('artifacts/dist/ai4sts2/LICENSE')]:
        archive.write(file, 'ai4sts2/' + file.name)
files = [Path('artifacts/dist/ai4sts2/ai4sts2.dll'), package]
Path('artifacts/dist/SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to("artifacts/dist")}\n' for p in files))
print(f'Package: {package}')
PY
