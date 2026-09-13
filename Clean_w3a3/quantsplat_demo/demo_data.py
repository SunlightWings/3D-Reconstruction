"""Read-only data access for the QuantSplat results explorer."""
from __future__ import annotations
import html
import json
import math
from pathlib import Path
from typing import Any

LABELS = {
    'gt': 'Ground truth', 'full': 'Full VGGT', 'w4a4': 'W4A4',
    'w3a3': 'W3A3', 'w3a3_conf': 'W3A3 + confidence',
    'w4a4_local': 'W4A4 (local calibration)', 'w4a4_rtn': 'W4A4 (RTN)',
}
METRICS = {'psnr': ('PSNR', 'dB', True), 'ssim': ('SSIM', '', True),
           'lpips': ('LPIPS', '', False)}


def escape(value: Any) -> str:
    return html.escape(str(value))


def number(value: Any, digits: int = 4) -> str:
    if value is None:
        return '\u2014'
    try:
        x = float(value)
        if not math.isfinite(x):
            return '\u2014'
        return f'{x:.{digits}f}'
    except (TypeError, ValueError):
        return '\u2014'


class DemoData:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.catalog = self.read('catalog.json', required=True)
        if self.catalog.get('schema_version') != 1:
            raise ValueError('Expected schema_version=1 in data/catalog.json. Run tools/pack_results.py.')
        self.scenes = self.catalog['scenes']
        if not self.scenes:
            raise ValueError('No scenes packaged. Run tools/pack_results.py first.')
        self.by_scene = {s['id']: s for s in self.scenes}
        self.models = {m['id']: m for m in self.catalog['models']}
        self.metrics = self.read('metrics.json')
        self.geometry = self.read('geometry.json')
        self.runtimes = self.read('runtime.json')
        self.rows = {row['scene']: row for row in self.metrics.get('per_scene', [])}

    def read(self, name: str, required: bool = False) -> dict:
        p = self.root / name
        if not p.is_file():
            if required:
                raise FileNotFoundError(f'Missing {p}; run tools/pack_results.py.')
            return {}
        with p.open(encoding='utf-8') as f:
            return json.load(f)

    def path(self, relative: str) -> Path:
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Asset path escapes the data folder.')
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    def label(self, arm: str) -> str:
        return self.models.get(arm, {}).get('label', LABELS.get(arm, arm))

    def image(self, scene_id: str, arm: str, index: int) -> str | None:
        s = self.by_scene[scene_id]
        paths = s['gt'] if arm == 'gt' else s.get('renders', {}).get(arm, [])
        if not paths:
            return None
        if len(paths) != len(s['frames']):
            raise ValueError(f'{scene_id}/{arm}: render count differs from GT.')
        return str(self.path(paths[max(0, min(index, len(paths) - 1))]))

    def availability(self, arm: str) -> int:
        return sum(bool(s.get('renders', {}).get(arm)) for s in self.scenes)

    def scene_metric(self, scene: str, arm: str, region: str, metric: str):
        return self.rows.get(scene, {}).get(f'{arm}_{region}_{metric}')

    def aggregate(self, arm: str, region: str, metric: str):
        return self.metrics.get('arms', {}).get(arm, {}).get(region, {}).get(metric)

    def reported_count(self, arm: str, region: str, metric: str) -> int:
        return sum(self.scene_metric(s['id'], arm, region, metric) is not None for s in self.scenes)

    def source_note(self) -> str:
        p = self.metrics.get('_provenance', {})
        if p.get('kind') == 'rounded_console_transcription':
            return ('Starter metrics were transcribed from your shared console output. '
                    'Per-scene values are rounded; summary and paired-test values are the reported values. '
                    'Replace them with your original results JSON before final submission.')
        return p.get('description', 'Metrics are imported from the supplied results JSON, not recalculated by this demo.')
