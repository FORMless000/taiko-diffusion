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
  apiBaseUrl: "https://ec2-18-117-249-161.us-east-2.compute.amazonaws.com"
};
```

This file is copied into the exported static site, so you can point the same build at a different backend host or port without changing application code.

## Build For FastAPI Static Hosting

```bash
cd webapp/frontend
npm run build
```

After the build writes `webapp/frontend/out/`, the app can be served from a plain static host. Job pages use `?job=<job_id>`, so the host does not need SPA rewrite support for `/jobs/<job_id>` paths.

## Local Launch Scripts

Convenience scripts are included under `webapp/scripts/`:

- `run-local.ps1`
- `run-local.sh`
- `run-backend-12205.ps1`
- `run-backend-12205.sh`
- `set-frontend-api-config.ps1`
- `set-frontend-api-config.sh`

They:

- write `webapp/frontend/public/config.js` with your chosen backend URL
- run the static frontend export build
- start the backend over plain HTTP
- start a plain HTTP static server for `webapp/frontend/out`

## AWS HTTPS Bridge

To bridge your local backend through the AWS FRP server with HTTPS:

1. Start the backend locally on `127.0.0.1:12205`.
2. Start `frpc` with `webapp/frpc.toml`.
3. Copy `webapp/aws/setup-https-bridge.sh` to the AWS Ubuntu/Debian server and run it with `sudo`.

The setup script:

- verifies `frps` is already running
- checks ports `443`, `80`, and `12205`
- installs Caddy if needed
- configures Caddy to request a normal certificate for `ec2-18-117-249-161.us-east-2.compute.amazonaws.com`
- reverse proxies `https://ec2-18-117-249-161.us-east-2.compute.amazonaws.com` to `http://127.0.0.1:12205`
- restarts Caddy and prints verification commands

Update the frontend runtime config with either helper script:

PowerShell:

```powershell
.\webapp\scripts\set-frontend-api-config.ps1 -ApiBaseUrl https://ec2-18-117-249-161.us-east-2.compute.amazonaws.com
```

Bash:

```bash
API_BASE_URL=https://ec2-18-117-249-161.us-east-2.compute.amazonaws.com ./webapp/scripts/set-frontend-api-config.sh
```

This hostname path only works if the EC2 DNS name is reachable publicly on ports 80 and 443 and Caddy can complete ACME validation.
