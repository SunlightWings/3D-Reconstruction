const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, { status = 0, detail = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function endpoint(path) {
  if (!API_BASE) throw new ApiError("The reconstruction service is not configured.");
  return `${API_BASE}${path}`;
}

function detailText(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item?.msg || String(item)).join("; ");
  return detail?.message || detail?.error || "";
}

async function responseJson(response, fallback) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = detailText(data.detail) || data.error || `${fallback} (${response.status})`;
    throw new ApiError(message, { status: response.status, detail: data });
  }
  return data;
}

async function request(path, options = {}, fallback = "Request failed") {
  try {
    const response = await fetch(endpoint(path), options);
    return await responseJson(response, fallback);
  } catch (error) {
    if (error?.name === "AbortError" || error instanceof ApiError) throw error;
    throw new ApiError("Unable to reach the reconstruction server.", { detail: String(error) });
  }
}

export function health(options = {}) {
  return request("/health", options, "Backend unavailable");
}

export function createJob(files, options = {}) {
  if (files.length !== 6) throw new ApiError("Exactly six images are required.");
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  return request("/api/jobs", { ...options, method: "POST", body: formData }, "Upload failed");
}

export function getJob(jobId, options = {}) {
  return request(`/api/jobs/${encodeURIComponent(jobId)}`, options, "Status request failed");
}

export async function pollJob(jobId, { onUpdate, signal, intervalMs = 2000 } = {}) {
  while (!signal?.aborted) {
    const job = await getJob(jobId, { signal });
    onUpdate?.(job);
    if (job.status === "complete" || job.status === "failed") return job;
    await new Promise((resolve, reject) => {
      const finish = () => { signal?.removeEventListener("abort", abort); resolve(); };
      const timer = window.setTimeout(finish, intervalMs);
      const abort = () => {
        window.clearTimeout(timer);
        reject(new DOMException("Polling cancelled", "AbortError"));
      };
      signal?.addEventListener("abort", abort, { once: true });
    });
  }
  throw new DOMException("Polling cancelled", "AbortError");
}

export const jobModelUrl = (job) => job?.result?.model_url || null;
