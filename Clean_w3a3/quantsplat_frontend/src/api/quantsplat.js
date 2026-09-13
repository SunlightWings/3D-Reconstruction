const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function endpoint(path) {
  if (!API_BASE) throw new Error("The reconstruction service is not configured.");
  return `${API_BASE}${path}`;
}

async function responseJson(response, fallback) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `${fallback} (${response.status})`);
  return data;
}

export async function health() {
  return responseJson(await fetch(endpoint("/health")), "Backend unavailable");
}

export async function createJob(files) {
  if (files.length !== 6) throw new Error("Please select exactly six images.");
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  return responseJson(await fetch(endpoint("/api/jobs"), { method: "POST", body: formData }), "Upload failed");
}

export async function getJob(jobId) {
  return responseJson(
    await fetch(endpoint(`/api/jobs/${encodeURIComponent(jobId)}`)),
    "Status request failed",
  );
}

export const jobModelUrl = (job) => job?.result?.model_url || null;
