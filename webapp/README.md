# Web App

This folder contains a deployable web interface for `taiko-diffusion`.

## Layout

- `backend/`: FastAPI API, single-worker job queue, filesystem job store.
- `frontend/`: Next.js UI that talks to the API and can be exported to static files.
- `runtime/jobs/`: local working directory for uploads, generated files, and job status JSON.

## Install

```bash
pip install -e .[webapp]
```

```bash
cd webapp/frontend
npm install
```

## Run In Development

Backend:

```bash
python -m uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd webapp/frontend
npm run build
python -m http.server 3000 --bind 127.0.0.1 --directory out
```

Update the static frontend runtime config before serving it:

```bash
webapp/frontend/public/config.js
```

Default contents:

```js
window.__TAIKO_CONFIG__ = {
  apiBaseUrl: "http://127.0.0.1:8000"
};
```

This file is copied into the exported static site, so you can point the same build at a different backend host or port without changing application code.

## Build For FastAPI Static Hosting

```bash
cd webapp/frontend
npm run build
```

After the build writes `webapp/frontend/out/`, the FastAPI backend serves that directory and falls back to `index.html` for client-side routes such as `/jobs/<job_id>`.

## Local Launch Scripts

Convenience scripts are included under `webapp/scripts/`:

- `run-local.ps1`
- `run-local.sh`

They:

- write `webapp/frontend/public/config.js` with your chosen backend URL
- run the static frontend export build
- start the backend over plain HTTP
- start a plain HTTP static server for `webapp/frontend/out`
