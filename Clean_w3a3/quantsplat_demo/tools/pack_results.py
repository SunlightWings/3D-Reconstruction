#!/usr/bin/env python3
"""Package existing PNG renders/JSONs for the demo. No training, inference, or rescoring.

Run --help for all paths. Camera IDs for local renders are checked against the
archive's frame-named GT; exactly nine PNGs alone is NOT sufficient validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo_data import LABELS

CORE_MODELS = [
    {'id': 'full', 'label': 'Full VGGT', 'description': 'Unquantized VGGT reference.'},
    {'id': 'w4a4', 'label': 'W4A4', 'description': '4-bit weight / activation quantization configuration.'},
    {'id': 'w3a3', 'label': 'W3A3', 'description': 'Locally calibrated 3-bit weight / activation configuration.'},
    {'id': 'w3a3_conf', 'label': 'W3A3 + confidence', 'description': 'Proposed lightweight confidence predictor; pending until results are supplied.'},
]
FRAME_RE = re.compile(r'^frame(\d+)\.png$')
SAFE_ID = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')


def read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


class Archive:
    """Read only the selected files from a directory or ZIP (no bulk extraction)."""
    def __init__(self, path: Path):
        self.path = path
        self.z = zipfile.ZipFile(path) if path.is_file() else None
        if self.z:
            self.names = [n for n in self.z.namelist() if not n.endswith('/')]
            # Allow one enclosing folder; identify the prefix using .../category/sequence/gt/frame.png.
            gt = [n for n in self.names if '/gt/frame' in n and n.endswith('.png')]
            if not gt:
                raise ValueError('No frame-named GT images found in the archive.')
            p = Path(gt[0]).parts
            self.prefix = '/'.join(p[:-4])
            self.prefix = self.prefix + '/' if self.prefix else ''
            self.relative = {n[len(self.prefix):]: n for n in self.names if n.startswith(self.prefix)}
        else:
            if not path.is_dir(): raise FileNotFoundError(path)
            self.relative = {p.relative_to(path).as_posix(): p for p in path.rglob('*.png')}

    def scenes(self):
        return sorted({'/'.join(Path(n).parts[:2]) for n in self.relative
                       if len(Path(n).parts) == 4 and Path(n).parts[2] == 'gt'
                       and FRAME_RE.fullmatch(Path(n).name)})

    def files(self, scene, arm):
        prefix = f'{scene}/{arm}/'
        return sorted(n for n in self.relative if n.startswith(prefix)
                      and len(Path(n).parts) == 4 and n.endswith('.png'))

    def arms(self, scene):
        return sorted({Path(n).parts[2] for n in self.relative
                       if n.startswith(scene + '/') and len(Path(n).parts) == 4
                       and Path(n).parts[2] != 'gt'})

    def copy(self, rel: str, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.z:
            with self.z.open(self.relative[rel]) as src, dest.open('wb') as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(self.relative[rel], dest)

    def close(self):
        if self.z: self.z.close()


def image_size(path):
    with Image.open(path) as im:
        im.load()
        if im.mode not in ('RGB', 'RGBA'):
            raise ValueError(f'{path}: expected RGB/RGBA image, got {im.mode}')
        return im.size


def read_binary_names(path: Path):
    def unpack(f: BinaryIO, fmt: str):
        size = struct.calcsize('<' + fmt); data = f.read(size)
        if len(data) != size: raise ValueError(f'Truncated COLMAP file: {path}')
        return struct.unpack('<' + fmt, data)
    names = []
    with path.open('rb') as f:
        (n,) = unpack(f, 'Q')
        if n > 10000: raise ValueError(f'Unexpected number of held-out images: {n}')
        for _ in range(n):
            unpack(f, 'i7di')
            name = bytearray()
            while True:
                b = f.read(1)
                if not b: raise ValueError(f'Truncated image name: {path}')
                if b == b'\0': break
                name.extend(b)
                if len(name) > 4096: raise ValueError('Invalid COLMAP image name.')
            names.append(name.decode('utf-8'))
            (points,) = unpack(f, 'Q')
            f.seek(24 * points, 1)
    return sorted(names)


def read_text_names(path: Path):
    lines = path.read_text(encoding='utf-8').splitlines()
    names = []; i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line or line.startswith('#'): continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10: raise ValueError(f'Invalid COLMAP image record in {path}: {line}')
        int(fields[0]); names.append(fields[9])
        # The following line stores observations (often blank for this export).
        if i < len(lines): i += 1
    return sorted(names)


def validate_local_order(scene_root, arm, expected_names, trust=False):
    sparse_root = scene_root / 'sources' / arm / 'heldout' / 'sparse'
    for sparse in [sparse_root / '0', sparse_root]:
        b = sparse / 'images.bin'; t = sparse / 'images.txt'
        names = read_binary_names(b) if b.is_file() else read_text_names(t) if t.is_file() else None
        if names is not None:
            names = [Path(n).name for n in names]
            if names != expected_names:
                raise ValueError(f'{scene_root.name}/{arm}: held-out camera names differ from the archive.\n'
                                 f'Archive: {expected_names}\nCamera source: {names}\n'
                                 'Do not select the first nine renders. Rebuild the matched held-out source.')
            return 'COLMAP held-out image names checked; sorted index pairing'
    if trust:
        return 'UNVERIFIED: user explicitly accepted sorted render order'
    raise FileNotFoundError(f'No held-out COLMAP images.txt/images.bin for {scene_root.name}/{arm}. '
                            'Supply the original source folder, or --trust-render-order only after verifying '
                            'the exact frame IDs and ordering independently.')


def select_group(root, scene):
    for p in sorted((root / scene).glob('*/full_meta.json')):
        m = read_json(p)
        if len(m.get('frame_numbers', [])) == 6 and len(m.get('primary_frame_numbers', [])) == 6 and not m.get('context_only_frame_numbers', []):
            return p.parent, m
    return None, None


def copy_metrics(source, old_data, stage, scene_ids):
    if source is None:
        source = old_data / 'metrics.json'
    if source.is_file():
        metrics = read_json(source)
        ids = {r['scene'] for r in metrics.get('per_scene', [])}
        if ids and ids != set(scene_ids):
            raise ValueError('The metrics JSON covers a different scene set from the render archive. '
                             'Use a results JSON for this exact cohort; do not display 40-scene averages for eight scenes.')
        if source.resolve() != (old_data / 'metrics.json').resolve():
            metrics['_provenance'] = {
                'kind': 'imported_results_json', 'filename': source.name,
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'description': 'Imported from the user-supplied results JSON. This app does not recalculate evaluation metrics.'}
        write_json(stage / 'metrics.json', metrics)
    else:
        write_json(stage / 'metrics.json', {'arms': {}, 'per_scene': [], 'pairs': {}})


def pack(args):
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.quantsplat-data-', dir=out.parent))
    archive = Archive(args.archive)
    try:
        scene_ids = archive.scenes()
        if not scene_ids: raise ValueError('No scenes found.')
        catalog = {'schema_version': 1, 'iterations': args.iterations, 'input_views': 6,
                   'models': list(CORE_MODELS), 'scenes': [],
                   'metric_policy': 'Imported raw foreground primary; content secondary; no live scoring.',
                   'packaged_at': datetime.now(timezone.utc).isoformat()}
        model_ids = {m['id'] for m in catalog['models']}
        runtime = {'records': [], 'notes': 'Imported raw timings, not a controlled speed benchmark.'}
        for sid in scene_ids:
            gt_files = archive.files(sid, 'gt')
            if len(gt_files) != 9: raise ValueError(f'{sid}: expected nine GT frames, got {len(gt_files)}.')
            names = [Path(n).name for n in gt_files]
            if not all(FRAME_RE.fullmatch(n) for n in names): raise ValueError(f'{sid}: GT names must be frameNNNNNN.png.')
            frames = [int(FRAME_RE.fullmatch(n).group(1)) for n in names]
            entry = {'id': sid, 'label': sid.split('/')[0].title(), 'frames': frames,
                     'gt': [], 'renders': {}, 'pairing': {}, 'inputs': []}
            sizes = []
            for src in gt_files:
                rel = f'images/{src}'; archive.copy(src, stage / rel)
                sizes.append(image_size(stage / rel)); entry['gt'].append(rel)
            if len(set(sizes)) != 1: raise ValueError(f'{sid}: inconsistent GT image sizes.')
            if sizes[0] != (518, 518): raise ValueError(f'{sid}: expected 518x518 images, got {sizes[0]}.')
            arms = archive.arms(sid) if args.all_archive_arms else args.archive_arms
            for arm in arms:
                if arm in args.local_arms: continue  # Never replace new results with archived variants.
                if not SAFE_ID.fullmatch(arm): raise ValueError(f'Invalid arm ID: {arm}')
                files = archive.files(sid, arm)
                if not files:
                    print(f'PENDING {sid}/{arm}: no archived images'); continue
                if [Path(n).name for n in files] != [f'{i:05d}.png' for i in range(9)]:
                    raise ValueError(f'{sid}/{arm}: expected exactly 00000.png through 00008.png.')
                entry['renders'][arm] = []
                for src, size in zip(files, sizes):
                    rel = f'images/{src}'; archive.copy(src, stage / rel)
                    if image_size(stage / rel) != size: raise ValueError(f'{src}: shape differs from GT.')
                    entry['renders'][arm].append(rel)
                entry['pairing'][arm] = 'Provided archive contract: sorted GT frame names correspond to sorted render indices.'
                if arm not in model_ids:
                    catalog['models'].append({'id': arm, 'label': LABELS.get(arm, arm), 'description': 'Archived experiment configuration.'}); model_ids.add(arm)
            local_scene = args.downstream / sid if args.downstream else None
            for arm in args.local_arms:
                if not SAFE_ID.fullmatch(arm): raise ValueError(f'Invalid arm ID: {arm}')
                rdir = local_scene / 'heldout' / f'{arm}_it{args.iterations}' / 'renders' if local_scene else None
                if rdir is None or not rdir.is_dir():
                    if args.require_local: raise FileNotFoundError(f'{sid}/{arm}: missing local renders: {rdir}')
                    print(f'PENDING {sid}/{arm}: no local images'); continue
                files = sorted(rdir.glob('*.png'))
                if [p.name for p in files] != [f'{i:05d}.png' for i in range(9)]:
                    raise ValueError(f'{sid}/{arm}: requires exactly nine indexed renders, found {len(files)}.')
                entry['pairing'][arm] = validate_local_order(local_scene, arm, names, args.trust_render_order)
                entry['renders'][arm] = []
                for src, size in zip(files, sizes):
                    rel = f'images/{sid}/{arm}/{src.name}'
                    dest = stage / rel; dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dest)
                    if image_size(dest) != size: raise ValueError(f'{src}: shape differs from GT.')
                    entry['renders'][arm].append(rel)
                # Check the local reference cache too, when present.
                for gp in entry['gt']:
                    local_gt = local_scene / 'common' / 'heldout_images' / Path(gp).name
                    if local_gt.is_file():
                        with Image.open(local_gt) as a, Image.open(stage / gp) as b:
                            if a.size != b.size or ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox():
                                raise ValueError(f'{sid}: local and archived GT pixels differ for {local_gt.name}.')
                if arm not in model_ids:
                    catalog['models'].append({'id': arm, 'label': LABELS.get(arm,arm), 'description': 'User-supplied experiment configuration.'}); model_ids.add(arm)
            if local_scene:
                inputs = sorted((local_scene / 'common' / 'train_images').glob('image_*.png'))
                if inputs and len(inputs) != 6: raise ValueError(f'{sid}: expected six common training images.')
                for src in inputs:
                    rel = f'images/{sid}/inputs/{src.name}'; dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(src,dest);image_size(dest);entry['inputs'].append(rel)
            if args.predictions:
                group, meta = select_group(args.predictions, sid)
                if group:
                    entry['input_frames'] = meta['frame_numbers']
                    if set(frames) & set(entry['input_frames']): raise ValueError(f'{sid}: input/evaluation frame overlap!')
                    for arm in ['full', 'w4a4'] + args.local_arms:
                        p = group / f'{arm}_meta.json'
                        if p.is_file():
                            m = read_json(p); rt = m.get('runtime', {})
                            runtime['records'].append({'scene':sid,'variant':arm, 'seconds':rt.get('seconds'),
                                                       'peak_allocated_mib':rt.get('peak_allocated_mib'),
                                                       'source':p.relative_to(args.predictions).as_posix()})
            catalog['scenes'].append(entry)
            print(f'OK {sid}: {len(frames)} views; {", ".join(entry["renders"])}')
        write_json(stage/'catalog.json', catalog)
        copy_metrics(args.metrics, out, stage, scene_ids)
        for name, supplied in [('geometry.json', args.geometry)]:
            src = supplied if supplied else out/name
            if src.is_file(): shutil.copyfile(src,stage/name)
        write_json(stage/'runtime.json',runtime if args.predictions else (read_json(out/'runtime.json') if (out/'runtime.json').is_file() else runtime))
        if args.check_only:
            print('CHECK PASSED. No packaged data was replaced.'); return
        backup = None
        if out.exists():
            backup = out.with_name(out.name + '_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            out.rename(backup)
        stage.rename(out)
        print(f'\nPACKED: {out}')
        if backup: print(f'Previous data preserved: {backup}')
        print('Restart the app, or upload the updated data/ folder to your Space. No NPZ/PLY/model checkpoints needed.')
    finally:
        archive.close()
        if stage.exists(): shutil.rmtree(stage)


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--archive',type=Path,required=True,help='renders7k_8scenes directory or ZIP.')
    p.add_argument('--downstream',type=Path,help='Root containing <category>/<sequence>/heldout/<arm>_it7000/renders/.')
    p.add_argument('--metrics',type=Path,help='Your original w3a3_vs_existing_8scenes_9views.json.')
    p.add_argument('--geometry',type=Path,help='Optional geometry_ablation_w3a3_subset.json.')
    p.add_argument('--predictions',type=Path,help='Optional prediction root for *_meta.json timing records.')
    p.add_argument('--archive-arms',nargs='+',default=['full','w4a4'])
    p.add_argument('--all-archive-arms',action='store_true',help='Also package available ablations; increases size.')
    p.add_argument('--local-arms',nargs='+',default=['w3a3'])
    p.add_argument('--iterations',type=int,default=7000)
    p.add_argument('--require-local',action='store_true',help='Fail if any requested local render set is absent.')
    p.add_argument('--trust-render-order',action='store_true',help='Explicitly allow missing COLMAP mapping; use only after independent verification.')
    p.add_argument('--out',type=Path,default=ROOT/'data')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    for name in ['metrics','geometry']:
        path=getattr(a,name)
        if path and not path.is_file(): p.error(f'{name} file does not exist: {path}')
    try: pack(a)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as ex:
        raise SystemExit(f'PACKING STOPPED: {ex}') from ex


if __name__=='__main__': main()
