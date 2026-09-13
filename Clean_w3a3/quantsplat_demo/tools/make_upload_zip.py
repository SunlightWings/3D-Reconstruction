#!/usr/bin/env python3
"""Build a clean upload archive, excluding environments, caches, and data backups."""
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/(ROOT.name+'_ready.zip')
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not p.is_file():continue
        if any(part in {'.git','.venv','__pycache__','.pytest_cache'} or part.startswith(('data_backup_','.quantsplat-data-')) for part in rel.parts):continue
        if p.suffix in {'.log','.pyc'}:continue
        z.write(p,rel.as_posix())
print(f'Created: {OUT}\nSize: {OUT.stat().st_size/2**20:.1f} MiB')
print('Unzip this archive and upload its CONTENTS, not the ZIP itself, to the Space repository root.')
