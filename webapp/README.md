# Web App

This folder contains the deployable web interface for `taiko-diffusion`.

It is intentionally narrower in scope than the top-level README: this document focuses on running, operating, and deploying the web app rather than explaining the whole research pipeline.

## Layout

- `backend/`: FastAPI API, single-worker job queue, and filesystem job store
- `frontend/`: Next.js frontend exported as static files
- `runtime/jobs/`: uploaded files, saved request payloads, job status JSON, and generated outputs
- `aws/`: deployment helpers such as the HTTPS bridge setup script

## Install

Python backend dependencies:

```bash
pip install -e .[webapp]
```

Frontend dependencies:

```bash
cd webapp/frontend
npm install
```

## Run Locally

### Backend

Run the backend:

```bash
python -m webapp.backend.main
```

This serves the API on `http://127.0.0.1:8000`.

### Frontend Build

Build the static frontend:

```bash
cd webapp/frontend
npm run build
```

The exported frontend is written to:

- `webapp/frontend/out/`

The backend can serve that exported site directly.

## Runtime Config

The exported frontend reads its runtime API host from:

- [webapp/frontend/out/config.js](C:/Users/28548/PythonNotebooks/taiko-diffusion/webapp/frontend/out/config.js)

The currently checked-in value points to:

- `https://ec2-18-117-249-161.us-east-2.compute.amazonaws.com`

If you deploy the app to another backend host, update the frontend config and rebuild or replace the exported assets.

## Model Registry Behavior

The frontend model dropdown is populated from:

- [webapp/backend/models.json](C:/Users/28548/PythonNotebooks/taiko-diffusion/webapp/backend/models.json)

The backend returns that manifest through:

- `GET /api/models`

If the manifest is missing or invalid, the backend falls back to the built-in model registry in:

- [src/inference/service.py](C:/Users/28548/PythonNotebooks/taiko-diffusion/src/inference/service.py)

Current model behavior to keep in mind:

- `sample_large_context` is the maxopt context checkpoint
- `sample_large_context_original` is the original context checkpoint
- several baseline AR checkpoints are also exposed
- diffusion-hybrid entries exist as separate selectable model options

## Current UI Behavior

The web UI accepts:

- required fields:
  - model
  - MP3
  - title
  - artist
  - difficulty name
  - BPM
  - offset
- advanced fields:
  - creator
  - meter
  - overall difficulty
  - density NPS
  - beatmap ID
  - temperature
  - source
  - tags

Operational notes:

- `beatmap_id` is used as a conditioning/auditing input for the model, not as ordinary user-facing beatmap metadata
- `temperature` is sent through sampling overrides
- inference currently assumes constant-BPM timing

## Job And Artifact Flow

Submitted jobs are written under:

- `webapp/runtime/jobs`

Each job stores:

- uploaded audio
- `request.json`
- `status.json`
- generated artifacts such as `.osu`, `.osz`, metadata JSON, timing JSON, notes JSON, and song-output JSON

Useful API routes:

- `GET /api/models`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/download/osz`

## Local Helper Scripts

Convenience scripts are included under `webapp/scripts/`:

- `run-local.ps1`
- `run-local.sh`
- `run-backend-12205.ps1`
- `run-backend-12205.sh`
- `set-frontend-api-config.ps1`
- `set-frontend-api-config.sh`

These help with:

- writing frontend runtime config
- rebuilding the static frontend
- starting the backend
- serving the exported frontend

## AWS HTTPS Bridge

For the current EC2 deployment flow, use:

- [webapp/aws/setup-https-bridge.sh](C:/Users/28548/PythonNotebooks/taiko-diffusion/webapp/aws/setup-https-bridge.sh)

That script is designed for a setup where:

- the backend is already being published through FRP
- an EC2 host terminates HTTPS with Caddy

The script:

- checks for an active `frps`
- installs Caddy if needed
- verifies that the local backend proxy port is reachable
- writes the Caddy reverse-proxy block
- configures firewall rules when `ufw` is active
- restarts Caddy and prints verification commands

Example usage on the EC2 host:

```bash
sudo PUBLIC_HOST=your.domain.example BACKEND_PROXY_PORT=12205 bash webapp/aws/setup-https-bridge.sh
```

Typical verification:

```bash
curl -v https://your.domain.example/api/models
sudo journalctl -u caddy -n 50 --no-pager
```

Deployment assumptions:

- DNS for the public host points to the EC2 instance
- AWS security groups allow ports `80` and `443`
- FRP is already configured to expose the backend on the chosen proxy port

## Limitation Reminder

The web app currently supports constant-BPM generation only.
