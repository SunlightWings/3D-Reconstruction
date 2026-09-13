# QuantSplat frontend

The production QuantSplat website: a static React/Vite client for the Modal reconstruction API
and the precomputed research explorer. It replaces the need to run `quantsplat_demo/app.py` in
production. The Gradio app remains in the repository only as a local legacy/reference viewer.

## Run locally

```bash
cp .env.example .env.local
npm install
npm run dev
```

The app accepts exactly six JPEG/PNG/WebP images, validates the 10 MiB per-image limit, creates
and resumes Modal jobs, polls every two seconds while processing, and persists the active job ID
in `localStorage`. The **Research explorer** section reads the existing active data pack in
`../quantsplat_demo/data/`; Vite copies it into the static deployment during `npm run build`.

The viewer activates only when a completed job supplies `result.model_url`. Until then, a completed geometry-only job presents an honest integration-pending state.

## Cloudflare Pages

Use this directory as the Pages root directory. Configure `npm run build` as the build command
and `dist` as the build output directory. The parent repository must be available during the
build because the Vite config copies `../quantsplat_demo/data` into the site. Set
`VITE_API_BASE_URL` for both Production and Preview; it is a public build-time URL, not a secret.

This first unified deployment includes the packaged PNG research data in the Pages build. If the
pack grows substantially, move `images/` and `turntable/` to Cloudflare R2 and set a public asset
base URL instead; the JSON metadata and React UI do not need to change conceptually.

Modal must allow the Pages URL and `http://localhost:5173` through CORS. Before public release, protect `POST /api/jobs` on the backend with server-verified Turnstile or equivalent abuse controls.
