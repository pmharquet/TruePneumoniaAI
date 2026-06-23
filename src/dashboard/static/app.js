const state = {
  project: null,
  status: null,
  socket: null,
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const number = Number(value);
  if (Math.abs(number) >= 100) return number.toFixed(1);
  if (Math.abs(number) >= 10) return number.toFixed(2);
  return number.toFixed(digits);
}

function metric(metrics, ...keys) {
  for (const key of keys) {
    if (metrics && metrics[key] !== undefined && metrics[key] !== null) return metrics[key];
  }
  return null;
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function loadProject() {
  state.project = await getJson("/api/project");
  renderProject();
  await loadSamples();
}

function renderProject() {
  const cfg = state.project.config;
  const dataSelect = $("dataDir");
  dataSelect.innerHTML = "";
  for (const dataset of state.project.datasets) {
    const option = document.createElement("option");
    option.value = dataset.name;
    option.textContent = `${dataset.name} (${dataset.total})`;
    option.disabled = !dataset.exists || dataset.total === 0;
    if (dataset.name === cfg.data.data_dir) option.selected = true;
    dataSelect.appendChild(option);
  }

  const form = $("trainForm");
  form.elements.backbone.value = cfg.model.backbone;
  form.elements.max_epochs.value = cfg.training.max_epochs;
  form.elements.batch_size.value = cfg.data.batch_size;
  form.elements.learning_rate.value = cfg.training.learning_rate;
  form.elements.num_workers.value = cfg.data.num_workers;
  form.elements.precision.value = cfg.training.precision;
  form.elements.log_every_n_steps.value = cfg.dashboard?.log_every_n_steps || 10;
  form.elements.pretrained.checked = Boolean(cfg.model.pretrained);

  renderDatasets();
  renderSystem(state.project.system);
  renderArtifacts(state.project.artifacts);
}

function renderDatasets() {
  const target = $("datasetSummary");
  target.innerHTML = "";
  for (const dataset of state.project.datasets) {
    const train = dataset.counts.train || {};
    const row = document.createElement("div");
    row.className = ROW;
    row.innerHTML = `
      <div class="min-w-0">
        <strong>${dataset.name}</strong>
        <div class="truncate text-[11px] text-slate-500">train N=${train.NORMAL || 0} P=${train.PNEUMONIA || 0}</div>
      </div>
      <strong>${dataset.total}</strong>
    `;
    target.appendChild(row);
  }
}

function renderSystem(system) {
  const target = $("systemInfo");
  const gpu = system.gpu?.length
    ? system.gpu.map((g) => g.name ? `${g.name} (${g.total_vram_gb} GB)` : g.error).join(", ")
    : "aucun GPU CUDA";
  const rows = [
    ["CUDA", system.cuda_available ? "oui" : "non"],
    ["GPU", gpu],
    ["Disque libre", `${system.disk.free_gb} / ${system.disk.total_gb} GB`],
    ["Python", system.python],
  ];
  target.innerHTML = rows.map(([k, v]) => `<div class="${ROW}"><span class="text-slate-500">${k}</span><strong class="break-words text-right">${v}</strong></div>`).join("");
}

function renderArtifacts(artifacts) {
  const target = $("artifacts");
  const checkpoints = artifacts.checkpoints || [];
  const rows = [];
  rows.push(`
    <div class="${ROW}">
      <span>ONNX</span>
      <strong>${artifacts.onnx.exists ? `${artifacts.onnx.size_mb} MB` : "absent"}</strong>
    </div>
  `);
  rows.push(`
    <div class="${ROW}">
      <span>MLflow</span>
      <strong>${artifacts.mlflow.exists ? "présent" : "absent"}</strong>
    </div>
  `);
  if (checkpoints.length === 0) {
    rows.push(`<div class="${ROW}"><span>Checkpoints</span><strong>aucun</strong></div>`);
  } else {
    for (const ckpt of checkpoints.slice(0, 8)) {
      rows.push(`
        <div class="${ROW}">
          <span class="truncate" title="${ckpt.path}">${ckpt.name}</span>
          <strong>${ckpt.size_mb} MB</strong>
        </div>
      `);
    }
  }
  const weights = artifacts.pretrained_weights || {};
  for (const [name, info] of Object.entries(weights)) {
    rows.push(`
      <div class="${ROW}">
        <span>Poids ${name}</span>
        <strong>${info.exists ? "présents" : "absents"}</strong>
      </div>
    `);
  }
  target.innerHTML = rows.join("");
}

async function loadSamples() {
  const dataset = $("dataDir").value || "chest_Xray_augmented";
  const data = await getJson(`/api/dataset/sample?dataset=${encodeURIComponent(dataset)}&split=train&class_name=NORMAL&limit=4`);
  const target = $("sampleGrid");
  target.innerHTML = "";
  for (const image of data.images) {
    const img = document.createElement("img");
    img.src = image.src;
    img.alt = image.name;
    img.title = image.name;
    img.className = "aspect-square w-full rounded-md border border-slate-200 bg-black object-cover";
    target.appendChild(img);
  }
}

const PILL_BASE = "inline-flex min-h-7 items-center rounded-full border px-2.5 text-[11px] font-extrabold uppercase ";
const PILL_VARIANTS = {
  running: "border-blue-200 bg-blue-50 text-blue-700",
  completed: "border-green-200 bg-green-50 text-green-700",
  failed: "border-red-200 bg-red-50 text-red-700",
  starting: "border-amber-200 bg-amber-50 text-amber-700",
};

function statusPillClass(status) {
  return PILL_BASE + (PILL_VARIANTS[status] || "border-slate-200 bg-slate-50 text-slate-500");
}

const ROW = "grid grid-cols-[1fr_auto] items-center gap-2 border-b border-slate-100 py-1.5 text-xs last:border-0";

function updateStatus(payload) {
  state.status = payload;
  const runState = payload.state || {};
  const status = payload.running ? "running" : (runState.status || "idle");
  const statusEl = $("runStatus");
  statusEl.className = statusPillClass(status);
  statusEl.textContent = status;

  $("startBtn").disabled = payload.running;
  $("stopBtn").disabled = !payload.running;
  $("runDir").textContent = payload.run_dir || "";

  renderProgress(runState);
  renderPerformance(runState.performance || {});
  renderMetrics(runState.latest_metrics || {});
  renderCharts(payload.events || []);
  renderArtifacts(payload.artifacts || {});
  renderSystem(payload.system || state.project?.system || {});
  $("logs").textContent = (payload.logs || []).join("\n");
}

function renderProgress(runState) {
  const epoch = Number(runState.epoch ?? 0);
  const maxEpochs = Number(runState.max_epochs ?? 0);
  const epochHuman = maxEpochs > 0 ? Math.min(epoch + 1, maxEpochs) : epoch;
  const trainBatch = Number(runState.train_batch ?? runState.val_batch ?? 0);
  const trainBatches = Number(runState.train_batches ?? runState.val_batches ?? 0);
  const epochPct = maxEpochs ? Math.min(100, (epochHuman / maxEpochs) * 100) : 0;
  const batchPct = trainBatches ? Math.min(100, (trainBatch / trainBatches) * 100) : 0;

  $("epochLabel").textContent = `${epochHuman} / ${maxEpochs || 0}`;
  $("batchLabel").textContent = `${trainBatch} / ${trainBatches || 0}`;
  $("epochBar").style.width = `${epochPct}%`;
  $("batchBar").style.width = `${batchPct}%`;
}

function fmtMem(mb) {
  if (mb === null || mb === undefined) return "-";
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "-";
  const s = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(sec).padStart(2, "0")}s`;
  return `${sec}s`;
}

function renderPerformance(perf) {
  const round0 = (v) => (v === null || v === undefined ? "-" : String(Math.round(Number(v))));
  $("perfImgS").textContent = round0(perf.img_per_s);
  $("perfImgSAvg").textContent = round0(perf.img_per_s_avg);
  $("perfMsBatch").textContent = round0(perf.ms_per_batch);
  $("perfStepsS").textContent = perf.steps_per_s != null ? fmt(perf.steps_per_s, 2) : "-";
  $("perfVram").textContent = fmtMem(perf.gpu_mem_mb);
  $("perfVramPeak").textContent = fmtMem(perf.gpu_mem_peak_mb);
  $("perfElapsed").textContent = fmtDuration(perf.elapsed_seconds);
  $("perfEta").textContent = fmtDuration(perf.eta_seconds);
}

function renderMetrics(metrics) {
  $("metricTrainLoss").textContent = fmt(metric(metrics, "train/loss", "train/loss_epoch"));
  $("metricTrainAuc").textContent = fmt(metric(metrics, "train/auroc"));
  $("metricValAuc").textContent = fmt(metric(metrics, "val/auroc"));
  $("metricSensitivity").textContent = fmt(metric(metrics, "val/sensitivity"));
  $("metricSpecificity").textContent = fmt(metric(metrics, "val/specificity"));
  $("metricF1").textContent = fmt(metric(metrics, "val/f1"));
  $("metricAccuracy").textContent = fmt(metric(metrics, "val/accuracy"));
  $("metricLr").textContent = fmt(metric(metrics, "lr"), 6);
}

function seriesFromEvents(events, keyNames) {
  const points = [];
  for (const event of events) {
    const metrics = event.metrics || {};
    const value = metric(metrics, ...keyNames);
    if (value !== null) {
      points.push({ x: Number(event.global_step ?? points.length), y: Number(value) });
    }
  }
  return points;
}

function drawChart(canvas, series, options = {}) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;
  const bufW = Math.round(width * dpr);
  const bufH = Math.round(height * dpr);
  if (canvas.width !== bufW) canvas.width = bufW;
  if (canvas.height !== bufH) canvas.height = bufH;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const all = series.flatMap((s) => s.points);
  if (all.length === 0) {
    ctx.fillStyle = "#647176";
    ctx.font = "14px system-ui";
    ctx.fillText("En attente de données", 24, 42);
    return;
  }

  const pad = { left: 52, right: 18, top: 18, bottom: 32 };
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  let minY = options.minY ?? Math.min(...ys);
  let maxY = options.maxY ?? Math.max(...ys);
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const sx = (x) => pad.left + ((x - minX) / Math.max(1, maxX - minX)) * plotW;
  const sy = (y) => pad.top + (1 - (y - minY) / Math.max(0.000001, maxY - minY)) * plotH;

  ctx.strokeStyle = "#e4eaed";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
  }
  ctx.stroke();

  ctx.fillStyle = "#647176";
  ctx.font = "11px system-ui";
  for (let i = 0; i <= 4; i++) {
    const value = maxY - ((maxY - minY) / 4) * i;
    ctx.fillText(fmt(value, 3), 8, pad.top + (plotH / 4) * i + 4);
  }

  for (const item of series) {
    if (item.points.length === 0) continue;
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    item.points.forEach((point, index) => {
      const x = sx(point.x);
      const y = sy(point.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  let legendX = pad.left;
  for (const item of series) {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, height - 20, 10, 10);
    ctx.fillStyle = "#172126";
    ctx.fillText(item.label, legendX + 14, height - 11);
    legendX += ctx.measureText(item.label).width + 40;
  }
}

function observeCharts() {
  if (typeof ResizeObserver === "undefined") {
    window.addEventListener("resize", () => renderCharts(state.lastEvents || []));
    return;
  }
  let scheduled = false;
  const observer = new ResizeObserver(() => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      renderCharts(state.lastEvents || []);
    });
  });
  for (const id of ["lossChart", "metricChart"]) {
    const canvas = $(id);
    if (canvas) observer.observe(canvas);
  }
}

function renderCharts(events) {
  state.lastEvents = events;
  const lossSeries = [
    { label: "train/loss", color: "#2563eb", points: seriesFromEvents(events, ["train/loss", "train/loss_epoch"]) },
    { label: "val/loss", color: "#a15c07", points: seriesFromEvents(events, ["val/loss"]) },
  ];
  drawChart($("lossChart"), lossSeries);

  const metricSeries = [
    { label: "val/auroc", color: "#12805c", points: seriesFromEvents(events, ["val/auroc"]) },
    { label: "sensitivity", color: "#2563eb", points: seriesFromEvents(events, ["val/sensitivity"]) },
    { label: "specificity", color: "#b42318", points: seriesFromEvents(events, ["val/specificity"]) },
  ];
  drawChart($("metricChart"), metricSeries, { minY: 0, maxY: 1 });
}

function formPayload() {
  const form = $("trainForm");
  return {
    data_dir: form.elements.data_dir.value,
    backbone: form.elements.backbone.value,
    pretrained: form.elements.pretrained.checked,
    max_epochs: Number(form.elements.max_epochs.value),
    batch_size: Number(form.elements.batch_size.value),
    learning_rate: Number(form.elements.learning_rate.value),
    num_workers: Number(form.elements.num_workers.value),
    precision: form.elements.precision.value,
    log_every_n_steps: Number(form.elements.log_every_n_steps.value),
  };
}

async function startTraining() {
  $("startBtn").disabled = true;
  try {
    await postJson("/api/train/start", formPayload());
    await refreshStatus();
  } catch (error) {
    alert(`Démarrage impossible: ${error.message}`);
  } finally {
    if (!state.status?.running) $("startBtn").disabled = false;
  }
}

async function stopTraining() {
  $("stopBtn").disabled = true;
  try {
    await postJson("/api/train/stop", {});
    await refreshStatus();
  } catch (error) {
    alert(`Arrêt impossible: ${error.message}`);
  }
}

async function refreshStatus() {
  updateStatus(await getJson("/api/status"));
}

function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/status`);
  state.socket = socket;
  socket.onmessage = (event) => updateStatus(JSON.parse(event.data));
  socket.onclose = () => {
    if (!state.pollTimer) state.pollTimer = setInterval(refreshStatus, 1500);
    setTimeout(connectSocket, 4000);
  };
}

async function boot() {
  $("startBtn").addEventListener("click", startTraining);
  $("stopBtn").addEventListener("click", stopTraining);
  $("refreshBtn").addEventListener("click", async () => {
    await loadProject();
    await refreshStatus();
  });
  $("dataDir").addEventListener("change", loadSamples);

  observeCharts();

  await loadProject();
  await refreshStatus();
  connectSocket();
}

boot().catch((error) => {
  console.error(error);
  $("logs").textContent = error.message;
});
