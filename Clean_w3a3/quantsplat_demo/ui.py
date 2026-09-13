"""HTML reporting components. No metric evaluation or model inference happens here."""
from __future__ import annotations
from demo_data import DemoData, escape as e, number as fmt, METRICS


def card(kicker, value, detail, tone=''):
    return f'<div class="stat {tone}"><div class="kicker">{e(kicker)}</div><strong>{e(value)}</strong><p>{e(detail)}</p></div>'


def hero(d: DemoData):
    views = sorted({len(s['frames']) for s in d.scenes})
    vc = str(views[0]) if len(views) == 1 else 'Variable'
    return '''<header class="hero">
      <div class="hero-top"><div class="brand"><span class="brand-mark">Q</span> QUANTSPLAT <span class="brand-slash">/</span> VISION LAB</div><span class="live-pill"><i></i> PRECOMPUTED EXPLORER</span></div>
      <div class="hero-main"><div><div class="eyebrow">PRECISION VS. RECONSTRUCTION</div>
      <h1>Fewer bits.<br><em>What changes?</em></h1>
      <p class="hero-copy">Six images. One reconstructed scene. Explore how quantizing VGGT changes the views produced by 3D Gaussian Splatting.</p></div>
      <div class="hero-aside"><span class="mini-label">THE EXPERIMENT</span><div class="model-pills"><b>Full</b><span>W4A4</span><span>W3A3</span></div><div class="method-pending">+ confidence predictor <span>IN PROGRESS</span></div><p>Compare the same camera view.<br>Inspect the evidence, not just the average.</p></div></div>
    </header>''' + '<div class="stat-grid">' + ''.join([
        card('SCENE COLLECTION', f"{len(d.scenes):02d}", 'CO3D evaluation scenes'),
        card('INPUT VIEWS', str(d.catalog.get('input_views', 6)).zfill(2), 'RGB images per reconstruction'),
        card('3DGS OPTIMIZATION', f"{d.catalog.get('iterations', 7000):,}", 'iterations per scene'),
        card('HELD-OUT VIEWS', vc.zfill(2), 'matched targets per scene'),
    ]) + '</div>'


def availability(d: DemoData):
    chips = []
    for arm in ['full', 'w4a4', 'w3a3', 'w3a3_conf']:
        n = d.availability(arm)
        status = f'{n}/{len(d.scenes)} scenes' if n else ('method pending' if arm == 'w3a3_conf' else 'images pending')
        chips.append(f'<span class="asset-chip {"ready" if n else "pending"}"><i></i>{e(d.label(arm))}<b>{e(status)}</b></span>')
    return '<div class="asset-strip">' + ''.join(chips) + '</div>'


def view_status(d, sid, idx, a, b):
    s = d.by_scene[sid]; idx = max(0, min(int(idx)-1, len(s['frames'])-1))
    absent = [d.label(m) for m in dict.fromkeys([a, b]) if not d.image(sid, m, idx)]
    status = f'<span class="warn-pill">Missing images: {e(", ".join(absent))}</span>' if absent else '<span class="ok-pill">Matched frame IDs</span>'
    return f'<div class="view-strip"><div><b>{e(s["label"])}</b><span>VIEW {idx+1:02d} / {len(s["frames"]):02d} &nbsp; &bull; &nbsp; GT FRAME {s["frames"][idx]:06d}</span></div>{status}</div>'


def table(headers, rows):
    return '<div class="table-wrap"><table class="qtable"><thead><tr>' + ''.join(f'<th>{e(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def scene_scores(d, sid, arms):
    rows = []
    for arm in dict.fromkeys(arms):
        if arm == 'gt':
            continue
        vals = [d.scene_metric(sid, arm, 'foreground', m) for m in METRICS]
        rows.append([f'<b>{e(d.label(arm))}</b>'] + [fmt(v) for v in vals])
    return '<div class="section-heading"><h3>Scene scores</h3><span>Foreground &middot; raw &middot; mean over all nine held-out views, not just this image</span></div>' + table(['Model', 'PSNR \u2191', 'SSIM \u2191', 'LPIPS \u2193'], rows) + '<p class="fineprint">A dash means no supplied score. Missing renders are never replaced with another model\u2019s output.</p>'


def summary_table(d, region='foreground', correction='Raw'):
    suffix = '' if correction == 'Raw' else '_exposure_corrected'
    rows = []
    for arm in d.models:
        vals = [d.aggregate(arm, region, m+suffix) for m in METRICS]
        n = d.reported_count(arm, region, 'psnr'+suffix)
        status = f'{n} scenes' if any(v is not None for v in vals) else 'Pending'
        rows.append([f'<b>{e(d.label(arm))}</b>', e(status)] + [fmt(v) for v in vals])
    return table(['Model', 'Coverage', 'PSNR \u2191 (dB)', 'SSIM \u2191', 'LPIPS \u2193'], rows)


def results_cards(d):
    pair = d.metrics.get('pairs', {}).get('full_minus_w3a3__foreground__psnr', {})
    if not pair:
        return '<div class="notice">Add the results JSON to display the measured Full\u2013W3A3 comparison.</div>'
    delta = pair.get('mean_delta'); ci = pair.get('ci95', [None, None]); n = pair.get('n', 0)
    return '<div class="result-stats">' + ''.join([
        card('REPORTED FULL - W3A3', f'{delta:+.4f} dB' if delta is not None else '\u2014', 'Raw foreground PSNR; supplied result', 'accent'),
        card('95% CONFIDENCE INTERVAL', f'{fmt(ci[0], 3)} to {fmt(ci[1], 3)}', 'Paired differences across scenes'),
        card('PAIRED SAMPLE', f'{n} scenes', 'Nine views averaged within each scene'),
    ]) + '</div>'


def pairs_table(d, region, correction):
    suffix = '' if correction == 'Raw' else '_exposure_corrected'
    rows=[]
    for arm in d.models:
        if arm=='full': continue
        for metric in METRICS:
            p=d.metrics.get('pairs',{}).get(f'full_minus_{arm}__{region}__{metric}{suffix}')
            if not p: continue
            ci=p.get('ci95', [None,None])
            rows.append([e(f'Full - {d.label(arm)}'), e(metric.upper()), e(p.get('n','?')),
                         fmt(p.get('mean_delta')), f'[{fmt(ci[0])}, {fmt(ci[1])}]', fmt(p.get('p'),5)])
    if not rows:
        return '<div class="notice">No paired statistics supplied for this selection. No p-values have been invented.</div>'
    return table(['Comparison', 'Metric', 'n', 'Mean difference', '95% CI', 'p (unadjusted)'], rows) + '<p class="fineprint">Full minus comparison model. Positive is better for Full on PSNR/SSIM; negative is better for Full on LPIPS. A non-significant test does not establish equivalence. These exploratory p-values are not adjusted for multiple comparisons.</p>'


def delta_plot(d, region, metric, correction):
    suffix='' if correction=='Raw' else '_exposure_corrected'
    active=[a for a in ['w4a4','w3a3','w3a3_conf'] if any(d.scene_metric(s['id'],a,region,metric+suffix) is not None for s in d.scenes)]
    values=[]
    for s in d.scenes:
        vals=[]
        f=d.scene_metric(s['id'],'full',region,metric+suffix)
        for a in active:
            q=d.scene_metric(s['id'],a,region,metric+suffix)
            vals.append(None if f is None or q is None else (f-q if metric!='lpips' else q-f))
        values.append(vals)
    finite=[x for xs in values for x in xs if x is not None]
    if not finite:
        return '<div class="notice">No per-scene values supplied for this metric.</div>'
    lo=min(0,min(finite)); hi=max(0,max(finite)); span=max(hi-lo,0.001);lo-=span*.13;hi+=span*.13
    left=138; plot_width=590; top=53; row_height=max(48,len(active)*14+15); height=top+len(values)*row_height+45
    scale=lambda x:left+(x-lo)/(hi-lo)*plot_width
    zero=scale(0)
    parts=[f'<div class="plot-shell"><svg viewBox="0 0 790 {height}" role="img" aria-label="Per-scene quality differences against Full">']
    parts += ['<text x="24" y="25" class="plot-title">Where does quality change?</text>']
    for j in range(5):
        val=lo+(hi-lo)*j/4;xx=scale(val)
        parts.append(f'<line x1="{xx:.2f}" y1="40" x2="{xx:.2f}" y2="{height-32}" class="grid-line"/><text x="{xx:.2f}" y="{height-10}" text-anchor="middle" class="tick">{val:+.3f}</text>')
    parts.append(f'<line x1="{zero:.2f}" y1="40" x2="{zero:.2f}" y2="{height-32}" class="zero-line"/>')
    for i,(s,vs) in enumerate(zip(d.scenes, values)):
        yy=top+i*row_height
        parts.append(f'<text x="124" y="{yy+13}" text-anchor="end" class="scene-tick">{e(s["label"])}</text>')
        for j,v in enumerate(vs):
            if v is None:continue
            xx=scale(v); arm=active[j];cl={'w4a4':'bar-teal','w3a3':'bar-coral','w3a3_conf':'bar-blue'}[arm]
            parts.append(f'<rect x="{min(zero,xx):.2f}" y="{yy+j*14}" width="{max(abs(xx-zero),1):.2f}" height="10" rx="2" class="{cl}"><title>{e(s["label"])} / {e(d.label(arm))}: {v:+.4f}</title></rect>')
    parts.append('</svg><div class="plot-legend">')
    for a in active:
        cl={'w4a4':'teal','w3a3':'coral','w3a3_conf':'blue'}[a]
        parts.append(f'<span><i class="{cl}"></i>{e(d.label(a))}</span>')
    parts.append('</div><p class="fineprint">Positive bars mean lower quality than Full. PSNR/SSIM: Full minus variant. LPIPS: variant minus Full. Values are per-scene averages; no new statistical tests are run here.</p></div>')
    return ''.join(parts)


def geometry_table(d):
    rows=[]
    for arm in d.models:
        block=d.geometry.get(arm,{}).get('mean',{})
        if not block:continue
        rows.append([e(d.label(arm)),e(block.get('n_scenes','Not supplied'))]+[fmt(block.get(k)) for k in ['cam_rot_deg','cam_centre_err','cam_spread_ratio','depth_rel_err','point_err']])
    if not rows:
        return '<div class="notice">Geometry diagnostics have not been added. Import geometry_ablation_w3a3_subset.json with the packer.</div>'
    return (('<div class="notice">Starter geometry values are transcribed from the reported eight-scene console summary. Import the original geometry JSON for full precision.</div>' if d.geometry.get('_provenance', {}).get('kind') == 'rounded_console_transcription' else '') + table(['Model','n','Rotation (deg)','Centre error','Spread ratio','Relative depth error','Point error'],rows))+'''<p class="fineprint">Errors are relative to Full, not ground-truth geometry. Camera-centre and point errors use Sim(3) alignment and normalization by the reference camera radius. The supplied spread ratio and depth error are unaligned and scale-sensitive; spread alone does not prove pose collapse or accuracy.</p>'''


def runtime_table(d):
    records=d.runtimes.get('records',[])
    if not records:
        return '<div class="notice">Timing data not packaged yet. The packer can read runtime.seconds and peak_allocated_mib from your *_meta.json files. No speedup is assumed from bit-width.</div>'
    rows=[]
    for r in records:
        rows.append([e(r['scene'].split('/')[0].title()),e(d.label(r['variant'])),fmt(r.get('seconds'),3),fmt(r.get('peak_allocated_mib'),1)])
    return table(['Scene','Model','Recorded inference (s)','Peak allocated (MiB)'],rows)+'''<p class="fineprint">Recorded per-scene timings only. First-call warm-up may be included. They exclude model loading, calibration, 3DGS optimization, and scoring unless explicitly measured by the upstream timer. Do not claim end-to-end speedup from this table; runs may differ in hardware and warm-up conditions.</p>'''


def data_status(d):
    rows=[]
    for s in d.scenes:
        rows.append([e(s['label']),str(len(s['frames']))]+[('Ready' if a in s.get('renders',{}) else 'Pending') for a in ['full','w4a4','w3a3','w3a3_conf']])
    return table(['Scene','GT views','Full','W4A4','W3A3','+ confidence'],rows)
