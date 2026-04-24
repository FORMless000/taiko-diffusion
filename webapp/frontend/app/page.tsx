"use client";

import { FormEvent, startTransition, useEffect, useState } from "react";

import { buildApiUrl } from "./config";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

type ModelDescriptor = {
  id: string;
  label: string;
  architecture_name: string;
  default_sampling: Record<string, unknown>;
  input_fields: Array<{
    id: string;
    label: string;
    kind: string;
    required: boolean;
    advanced: boolean;
  }>;
  output_artifact_kinds: string[];
  enabled: boolean;
};

type JobArtifact = {
  id: string;
  kind: string;
  label: string;
  relative_path: string;
  media_type: string;
  primary_download: boolean;
};

type JobResponse = {
  job_id: string;
  model_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string;
  artifacts: JobArtifact[];
  primary_download_url: string | null;
};

type FormState = {
  modelId: string;
  title: string;
  artist: string;
  version: string;
  bpm: string;
  offsetMs: string;
  creator: string;
  meter: string;
  overallDifficulty: string;
  densityNps: string;
  source: string;
  tags: string;
};

const DEFAULT_FORM: FormState = {
  modelId: "",
  title: "",
  artist: "",
  version: "",
  bpm: "",
  offsetMs: "0",
  creator: "taiko-diffusion",
  meter: "4",
  overallDifficulty: "",
  densityNps: "6.0",
  source: "",
  tags: "",
};

function withBasePath(pathname: string): string {
  const normalized = pathname.startsWith("/") ? pathname : `/${pathname}`;
  const basePath = getBasePath(typeof window === "undefined" ? "/" : window.location.pathname);
  if (!basePath) {
    return normalized;
  }
  if (normalized === "/") {
    return `${basePath}/`;
  }
  return `${basePath}${normalized}`;
}

function getBasePath(pathname: string): string {
  if (BASE_PATH) {
    return BASE_PATH;
  }
  const normalized = pathname.replace(/\/+$/, "") || "/";
  if (normalized === "/") {
    return "";
  }
  return normalized;
}

function getJobIdFromSearch(search: string): string | null {
  const params = new URLSearchParams(search);
  const jobId = params.get("job");
  return jobId ? jobId.trim() || null : null;
}

function buildAppUrl(jobId: string | null): string {
  const pathname = withBasePath("/");
  if (!jobId) {
    return pathname;
  }
  const url = new URL(pathname, "https://example.invalid");
  url.searchParams.set("job", jobId);
  return `${url.pathname}${url.search}`;
}

function formatTimestamp(value: string | null): string {
  if (!value) {
    return "—";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function Field({
  label,
  value,
  onChange,
  type = "text",
  required = false,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (nextValue: string) => void;
  type?: "text" | "number";
  required?: boolean;
  placeholder?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type={type}
        required={required}
        step={type === "number" ? "any" : undefined}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function HomeScreen({
  models,
  form,
  audioFile,
  submitError,
  submitting,
  onFormChange,
  onAudioChange,
  onSubmit,
}: {
  models: ModelDescriptor[];
  form: FormState;
  audioFile: File | null;
  submitError: string;
  submitting: boolean;
  onFormChange: (key: keyof FormState, value: string) => void;
  onAudioChange: (file: File | null) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => Promise<void>;
}) {
  const enabledModels = models.filter((model) => model.enabled);
  const selectedModel = enabledModels.find((model) => model.id === form.modelId) ?? enabledModels[0] ?? null;

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">Taiko Diffusion</p>
        <h1>Generate a taiko map, keep the flow simple.</h1>
        <p className="lede">
          Upload one MP3, fill the minimal metadata and timing, pick a model, then download a generated <code>.osz</code>.
        </p>
      </section>

      <section className="panel">
        <form className="stack" onSubmit={onSubmit}>
          <label className="field">
            <span>Model</span>
            <select
              required
              value={form.modelId}
              onChange={(event) => onFormChange("modelId", event.target.value)}
            >
              {enabledModels.length === 0 ? <option value="">No checkpoints available</option> : null}
              {enabledModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>MP3 File</span>
            <input
              type="file"
              accept=".mp3,audio/mpeg"
              required
              onChange={(event) => onAudioChange(event.target.files?.[0] ?? null)}
            />
          </label>

          <div className="grid">
            <Field label="Title" required value={form.title} onChange={(value) => onFormChange("title", value)} />
            <Field label="Artist" required value={form.artist} onChange={(value) => onFormChange("artist", value)} />
            <Field
              label="Difficulty Name"
              required
              value={form.version}
              onChange={(value) => onFormChange("version", value)}
              placeholder="Oni"
            />
            <Field label="BPM" required type="number" value={form.bpm} onChange={(value) => onFormChange("bpm", value)} />
            <Field
              label="Offset (ms)"
              required
              type="number"
              value={form.offsetMs}
              onChange={(value) => onFormChange("offsetMs", value)}
            />
          </div>

          <details className="details">
            <summary>Advanced</summary>
            <div className="grid">
              <Field label="Creator" value={form.creator} onChange={(value) => onFormChange("creator", value)} />
              <Field label="Meter" type="number" value={form.meter} onChange={(value) => onFormChange("meter", value)} />
              <Field
                label="Overall Difficulty"
                type="number"
                value={form.overallDifficulty}
                onChange={(value) => onFormChange("overallDifficulty", value)}
              />
              <Field
                label="Density NPS"
                type="number"
                value={form.densityNps}
                onChange={(value) => onFormChange("densityNps", value)}
              />
              <Field label="Source" value={form.source} onChange={(value) => onFormChange("source", value)} />
              <Field label="Tags" value={form.tags} onChange={(value) => onFormChange("tags", value)} />
            </div>
          </details>

          {selectedModel ? (
            <div className="modelCard">
              <div>
                <strong>{selectedModel.label}</strong>
                <p>{selectedModel.architecture_name}</p>
              </div>
              <p className="modelMeta">
                Outputs: {selectedModel.output_artifact_kinds.join(", ")}
              </p>
            </div>
          ) : null}

          {audioFile ? <p className="hint">Selected audio: {audioFile.name}</p> : null}
          {submitError ? <p className="error">{submitError}</p> : null}

          <button className="primaryButton" type="submit" disabled={submitting || enabledModels.length === 0}>
            {submitting ? "Submitting..." : "Generate"}
          </button>
        </form>
      </section>
    </main>
  );
}

function JobScreen({
  jobId,
  job,
  loading,
  loadError,
  onBack,
}: {
  jobId: string;
  job: JobResponse | null;
  loading: boolean;
  loadError: string;
  onBack: () => void;
}) {
  return (
    <main className="shell">
      <section className="hero compact">
        <p className="eyebrow">Generation Job</p>
        <h1>Track progress, then grab the archive.</h1>
        <p className="lede">Job ID: <code>{jobId}</code></p>
      </section>

      <section className="panel stack">
        <div className="statusRow">
          <div>
            <span className={`statusPill ${job?.status ?? "queued"}`}>{job?.status ?? "loading"}</span>
            <p className="hint">Created: {formatTimestamp(job?.created_at ?? null)}</p>
          </div>
          <button className="ghostButton" type="button" onClick={onBack}>
            New Generation
          </button>
        </div>

        {loading && !job ? <p className="hint">Loading job status...</p> : null}
        {loadError ? <p className="error">{loadError}</p> : null}

        {job ? (
          <>
            <div className="grid statusGrid">
              <div className="metric">
                <span>Started</span>
                <strong>{formatTimestamp(job.started_at)}</strong>
              </div>
              <div className="metric">
                <span>Finished</span>
                <strong>{formatTimestamp(job.finished_at)}</strong>
              </div>
              <div className="metric">
                <span>Model</span>
                <strong>{job.model_id}</strong>
              </div>
            </div>

            {job.error ? <p className="error">{job.error}</p> : null}

            {job.status === "succeeded" && job.primary_download_url ? (
              <a className="primaryButton downloadLink" href={buildApiUrl(job.primary_download_url)}>
                Download .osz
              </a>
            ) : null}

            <details className="details">
              <summary>Debug Artifacts</summary>
              <div className="artifactList">
                {job.artifacts.length === 0 ? (
                  <p className="hint">No artifacts yet.</p>
                ) : (
                  job.artifacts.map((artifact) => (
                    <div className="artifactItem" key={artifact.id}>
                      <strong>{artifact.label}</strong>
                      <code>{artifact.relative_path}</code>
                    </div>
                  ))
                )}
              </div>
            </details>
          </>
        ) : null}
      </section>
    </main>
  );
}

export default function Page() {
  const [models, setModels] = useState<ModelDescriptor[]>([]);
  const [modelsError, setModelsError] = useState("");
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [currentSearch, setCurrentSearch] = useState("");
  const [job, setJob] = useState<JobResponse | null>(null);
  const [jobLoading, setJobLoading] = useState(false);
  const [jobError, setJobError] = useState("");

  const currentJobId = getJobIdFromSearch(currentSearch);

  useEffect(() => {
    const syncLocation = () => setCurrentSearch(window.location.search);
    syncLocation();
    window.addEventListener("popstate", syncLocation);
    return () => window.removeEventListener("popstate", syncLocation);
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch(buildApiUrl("/api/models"))
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Model request failed with status ${response.status}`);
        }
        return response.json();
      })
      .then((payload: { models: ModelDescriptor[] }) => {
        if (cancelled) {
          return;
        }
        const nextModels = payload.models ?? [];
        setModels(nextModels);
        setModelsError("");
        setForm((current) => ({
          ...current,
          modelId: current.modelId || nextModels.find((model) => model.enabled)?.id || "",
        }));
      })
      .catch((error: Error) => {
        if (cancelled) {
          return;
        }
        setModelsError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!currentJobId) {
      return;
    }

    let cancelled = false;
    let timeoutId: number | null = null;

    const loadJob = async () => {
      setJobLoading(true);
      try {
        const response = await fetch(buildApiUrl(`/api/jobs/${currentJobId}`));
        if (!response.ok) {
          throw new Error(`Job request failed with status ${response.status}`);
        }
        const payload = (await response.json()) as JobResponse;
        if (cancelled) {
          return;
        }
        setJob(payload);
        setJobError("");
        if (payload.status === "queued" || payload.status === "running") {
          timeoutId = window.setTimeout(loadJob, 2000);
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        setJobError(error instanceof Error ? error.message : "Could not load job status.");
      } finally {
        if (!cancelled) {
          setJobLoading(false);
        }
      }
    };

    loadJob();
    return () => {
      cancelled = true;
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [currentJobId]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitError("");

    if (!audioFile) {
      setSubmitError("Choose an MP3 file before generating.");
      return;
    }

    setSubmitting(true);
    try {
      const metadata = {
        title: form.title,
        artist: form.artist,
        version: form.version,
        creator: form.creator || "taiko-diffusion",
        source: form.source,
        tags: form.tags,
        overall_difficulty: form.overallDifficulty ? Number(form.overallDifficulty) : null,
      };
      const timing = {
        bpm: Number(form.bpm),
        offset_ms: Number(form.offsetMs),
        meter: Number(form.meter || "4"),
      };
      const conditioning = {
        density_nps: Number(form.densityNps || "6.0"),
        difficulty_value: form.overallDifficulty ? Number(form.overallDifficulty) : null,
      };

      const requestBody = new FormData();
      requestBody.append("model_id", form.modelId);
      requestBody.append("metadata_json", JSON.stringify(metadata));
      requestBody.append("timing_json", JSON.stringify(timing));
      requestBody.append("conditioning_json", JSON.stringify(conditioning));
      requestBody.append("sampling_override_json", JSON.stringify({}));
      requestBody.append("audio_file", audioFile);

      const response = await fetch(buildApiUrl("/api/jobs"), {
        method: "POST",
        body: requestBody,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Job submission failed with status ${response.status}`);
      }
      const payload = (await response.json()) as { job_id: string };
      startTransition(() => {
        const nextUrl = buildAppUrl(payload.job_id);
        window.history.pushState({}, "", nextUrl);
        setCurrentSearch(new URL(nextUrl, window.location.origin).search);
        setJob(null);
        setJobError("");
      });
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Generation request failed.");
    } finally {
      setSubmitting(false);
    }
  }

  function handleBack() {
    startTransition(() => {
      const homeUrl = buildAppUrl(null);
      window.history.pushState({}, "", homeUrl);
      setCurrentSearch("");
      setJob(null);
      setJobError("");
    });
  }

  if (currentJobId) {
    return (
      <JobScreen
        jobId={currentJobId}
        job={job}
        loading={jobLoading}
        loadError={jobError}
        onBack={handleBack}
      />
    );
  }

  return (
    <>
      {modelsError ? <p className="bannerError">{modelsError}</p> : null}
      <HomeScreen
        models={models}
        form={form}
        audioFile={audioFile}
        submitError={submitError}
        submitting={submitting}
        onFormChange={(key, value) => setForm((current) => ({ ...current, [key]: value }))}
        onAudioChange={setAudioFile}
        onSubmit={handleSubmit}
      />
    </>
  );
}
