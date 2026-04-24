declare global {
  interface Window {
    __TAIKO_CONFIG__?: {
      apiBaseUrl?: string;
    };
    TAIKO_WEBAPP_CONFIG?: {
      apiBaseUrl?: string;
    };
  }
}

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

function normalizeApiBaseUrl(value: string): string {
  const trimmed = String(value || "").trim();
  if (!trimmed) {
    return DEFAULT_API_BASE_URL;
  }
  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed.replace(/\/+$/, "");
  }
  if (/^[a-z0-9.-]+:\d+(\/.*)?$/i.test(trimmed) || /^localhost:\d+(\/.*)?$/i.test(trimmed)) {
    return `http://${trimmed}`.replace(/\/+$/, "");
  }
  return trimmed.replace(/\/+$/, "");
}

export function getApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return DEFAULT_API_BASE_URL;
  }

  const configured =
    window.__TAIKO_CONFIG__?.apiBaseUrl ||
    window.TAIKO_WEBAPP_CONFIG?.apiBaseUrl ||
    DEFAULT_API_BASE_URL;
  return normalizeApiBaseUrl(configured);
}

export function buildApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${getApiBaseUrl()}${normalizedPath}`;
}
