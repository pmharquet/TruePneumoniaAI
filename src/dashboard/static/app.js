const state = {
  project: null,
  status: null,
  socket: null,
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

// Classes a dataset holds, as [negative, positive]. The backend declares them
// per dataset (NORMAL/PNEUMONIA for the binary task, VIRUS/BACTERIA for the
// pneumonia-subtype task). Falls back to the binary pair if unknown.
function datasetClasses(name) {
  const datasets = state.project?.datasets || [];
  const match = datasets.find((d) => d.name === name);
  return match?.classes?.length === 2 ? match.classes : ["NORMAL", "PNEUMONIA"];
}

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
    const classes = dataset.classes?.length === 2 ? dataset.classes : ["NORMAL", "PNEUMONIA"];
    const summary = classes
      .map((c) => `${c[0]}=${train[c] || 0}`)
      .join(" ");
    const row = document.createElement("div");
    row.className = ROW;
    row.innerHTML = `
      <div class="min-w-0">
        <strong>${dataset.name}</strong>
        <div class="truncate text-[11px] text-slate-500">train ${summary}</div>
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
  const className = datasetClasses(dataset)[0];
  const data = await getJson(`/api/dataset/sample?dataset=${encodeURIComponent(dataset)}&split=train&class_name=${className}&limit=4`);
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
  renderTestMetrics(runState);
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

function renderTestMetrics(runState) {
  const m = runState.test_metrics;
  const section = $("testMetricsSection");
  if (!m || Object.keys(m).length === 0) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");
  $("testAuroc").textContent = fmt(metric(m, "test/auroc"), 3);
  $("testSens").textContent = fmt(metric(m, "test/sensitivity"), 3);
  $("testSpec").textContent = fmt(metric(m, "test/specificity"), 3);
  $("testF1").textContent = fmt(metric(m, "test/f1"), 3);
  $("testAcc").textContent = fmt(metric(m, "test/accuracy"), 3);
  const thr = runState.test_threshold;
  $("testThreshold").textContent = thr != null ? `seuil ${Number(thr).toFixed(2)} · test complet` : "test complet";
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

  // Test curves (per epoch) — val saturates near 1.0 and is uninformative.
  const metricSeries = [
    { label: "test/accuracy", color: "#12805c", points: seriesFromEvents(events, ["test/accuracy"]) },
    { label: "test/specificity", color: "#b42318", points: seriesFromEvents(events, ["test/specificity"]) },
    { label: "test/auroc", color: "#2563eb", points: seriesFromEvents(events, ["test/auroc"]) },
  ];
  // Auto-zoom: fit the axis to the data range (with padding) so the variation
  // is visible instead of being flattened against a fixed 0..1 axis.
  drawChart($("metricChart"), metricSeries, paddedRange(metricSeries));
}

function paddedRange(seriesList, { floor = 0, ceil = 1, pad = 0.02 } = {}) {
  const ys = seriesList.flatMap((s) => s.points.map((p) => p.y));
  if (ys.length === 0) return { minY: 0, maxY: 1 };
  let lo = Math.max(floor, Math.min(...ys) - pad);
  let hi = Math.min(ceil, Math.max(...ys) + pad);
  if (hi - lo < 0.04) {
    lo = Math.max(floor, lo - 0.03);
    hi = Math.min(ceil, hi + 0.03);
  }
  return { minY: lo, maxY: hi };
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

// ===================== Test des modèles =====================

const TAB_ACTIVE = "bg-white text-slate-900 shadow-sm";
const TAB_INACTIVE = "text-slate-600";

function setView(view) {
  $("view-train").classList.toggle("hidden", view !== "train");
  $("view-train").classList.toggle("grid", view === "train");
  $("view-test").classList.toggle("hidden", view !== "test");
  $("view-test").classList.toggle("grid", view === "test");
  for (const btn of document.querySelectorAll(".tab-btn")) {
    const active = btn.dataset.view === view;
    btn.className = `tab-btn min-h-7 rounded px-3 ${active ? TAB_ACTIVE : TAB_INACTIVE}`;
  }
  if (view === "test") {
    if (!state.testLoaded) {
      initTestPage();
    } else {
      // refresh on each visit — checkpoints change during training, and the
      // dataset list may have grown (e.g. a freshly generated split).
      populateTestDatasets();
      loadModels();
    }
  }
}

async function initTestPage() {
  state.testLoaded = true;
  await populateTestDatasets();
  updatePredClasses();
  await loadModels();
  await loadPredSamples();
}

// Fill the "true class" picker with the selected dataset's classes so the
// subtype dataset offers BACTERIA/VIRUS instead of NORMAL/PNEUMONIA.
function updatePredClasses() {
  const select = $("predClass");
  if (!select) return;
  const previous = select.value;
  const classes = datasetClasses($("testDataset").value);
  select.innerHTML = "";
  for (const name of classes) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }
  if (classes.includes(previous)) select.value = previous;
}

async function populateTestDatasets() {
  const datasetSelect = $("testDataset");
  // Don't rely on boot-time state.project (it may not be loaded yet on the
  // first tab open, or the dataset list may have grown) — fetch fresh.
  let datasets = state.project?.datasets;
  if (!datasets || datasets.length === 0) {
    try {
      const project = await getJson("/api/project");
      state.project = project;
      datasets = project.datasets;
    } catch (error) {
      datasets = [];
    }
  }
  const previous = datasetSelect.value;
  datasetSelect.innerHTML = "";
  let firstEnabled = null;
  for (const dataset of datasets || []) {
    const option = document.createElement("option");
    option.value = dataset.name;
    option.textContent = `${dataset.name} (${dataset.total})`;
    option.disabled = !dataset.exists || dataset.total === 0;
    if (!option.disabled && firstEnabled === null) firstEnabled = dataset.name;
    datasetSelect.appendChild(option);
  }
  // Default to the first available dataset (chest_Xray_patient, the leak-free
  // split) rather than the training data_dir — keep the user's choice if still
  // valid across refreshes.
  const options = [...datasetSelect.options];
  const keep = options.find((o) => o.value === previous && !o.disabled);
  datasetSelect.value = keep ? previous : (firstEnabled || datasetSelect.value);
}

async function loadModels() {
  const select = $("modelSelect");
  try {
    const data = await getJson("/api/models");
    select.innerHTML = "";
    const checkpoints = data.checkpoints || [];
    if (checkpoints.length === 0) {
      select.innerHTML = `<option value="">aucun checkpoint</option>`;
      return;
    }
    for (const ckpt of checkpoints) {
      const option = document.createElement("option");
      option.value = ckpt.path;
      option.textContent = `${ckpt.name} (${ckpt.size_mb} MB)`;
      select.appendChild(option);
    }
  } catch (error) {
    select.innerHTML = `<option value="">erreur: ${error.message}</option>`;
  }
}

function currentThreshold() {
  return Number($("thresholdSlider").value);
}

async function runEvaluate() {
  const checkpoint = $("modelSelect").value;
  if (!checkpoint) {
    alert("Sélectionne un checkpoint.");
    return;
  }
  const btn = $("evalBtn");
  btn.disabled = true;
  $("evalMeta").textContent = "évaluation en cours…";
  try {
    const result = await postJson("/api/evaluate", {
      checkpoint,
      dataset: $("testDataset").value,
      split: $("evalSplit").value,
      threshold: currentThreshold(),
    });
    const m = result.metrics;
    $("evalAuroc").textContent = fmt(m.auroc, 3);
    $("evalSens").textContent = fmt(m.sensitivity, 3);
    $("evalSpec").textContent = fmt(m.specificity, 3);
    $("evalF1").textContent = fmt(m.f1, 3);
    $("evalAcc").textContent = fmt(m.accuracy, 3);
    const cls = result.classes || ["NORMAL", "PNEUMONIA"];
    const nNeg = result.n_negative ?? result.n_normal;
    const nPos = result.n_positive ?? result.n_pneumonia;
    $("evalMeta").textContent = `${result.count} images · ${cls[0]}=${nNeg} ${cls[1]}=${nPos} · seuil ${result.threshold.toFixed(2)}`;
    const c = result.confusion;
    const cells = [
      ["Vrais positifs", c.tp, "text-green-700"],
      ["Vrais négatifs", c.tn, "text-green-700"],
      ["Faux positifs", c.fp, "text-red-700"],
      ["Faux négatifs", c.fn, "text-red-700"],
    ];
    $("evalConfusion").innerHTML = cells
      .map(([label, value, color]) => `<div class="rounded-md border border-slate-200 px-2.5 py-2"><span class="block text-[10px] font-bold uppercase tracking-wide text-slate-500">${label}</span><strong class="mt-1 block text-base tabular-nums ${color}">${value}</strong></div>`)
      .join("");
  } catch (error) {
    $("evalMeta").textContent = "";
    if (/not found/i.test(error.message)) {
      await loadModels();
      alert("Ce checkpoint n'existe plus (remplacé pendant l'entraînement). Liste rafraîchie — resélectionne un modèle.");
    } else {
      alert(`Évaluation impossible: ${error.message}`);
    }
  } finally {
    btn.disabled = false;
  }
}

async function loadPredSamples() {
  const dataset = $("testDataset").value || "chest_Xray_augmented";
  const split = $("predSplit").value;
  const className = $("predClass").value;
  const grid = $("predGrid");
  grid.innerHTML = `<span class="col-span-full text-[11px] text-slate-400">chargement…</span>`;
  try {
    const data = await getJson(`/api/dataset/sample?dataset=${encodeURIComponent(dataset)}&split=${split}&class_name=${className}&limit=8`);
    grid.innerHTML = "";
    for (const image of data.images) {
      const img = document.createElement("img");
      img.src = image.src;
      img.alt = image.name;
      img.title = `${image.name} — cliquer pour prédire`;
      img.className = "aspect-square w-full cursor-pointer rounded-md border border-slate-200 bg-black object-cover hover:ring-2 hover:ring-blue-400";
      img.addEventListener("click", () => predictSample(image.name, image.src, className));
      grid.appendChild(img);
    }
    if (data.images.length === 0) grid.innerHTML = `<span class="col-span-full text-[11px] text-slate-400">aucune image</span>`;
  } catch (error) {
    grid.innerHTML = `<span class="col-span-full text-[11px] text-red-600">${error.message}</span>`;
  }
}

async function predictSample(name, src, trueLabel) {
  const checkpoint = $("modelSelect").value;
  if (!checkpoint) {
    alert("Sélectionne un checkpoint.");
    return;
  }
  try {
    const result = await postJson("/api/predict", {
      checkpoint,
      dataset: $("testDataset").value,
      split: $("predSplit").value,
      class_name: trueLabel,
      name,
      threshold: currentThreshold(),
    });
    renderPrediction(result, src, trueLabel);
  } catch (error) {
    alert(`Prédiction impossible: ${error.message}`);
  }
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function predictUpload(file) {
  const checkpoint = $("modelSelect").value;
  if (!checkpoint) {
    alert("Sélectionne un checkpoint.");
    return;
  }
  try {
    const dataUrl = await readFileAsDataURL(file);
    const result = await postJson("/api/predict-upload", {
      checkpoint,
      threshold: currentThreshold(),
      image: dataUrl,
    });
    renderPrediction(result, dataUrl, null);
  } catch (error) {
    alert(`Prédiction impossible: ${error.message}`);
  }
}

function renderPrediction(result, src, trueLabel) {
  const box = $("predResult");
  box.classList.remove("hidden");
  box.classList.add("grid");
  $("predImage").src = src;

  const prob = Number(result.probability);
  // prob is P(positive class); colour red when the prediction is the positive
  // (clinically flagged) class — PNEUMONIA or, for the subtype model, BACTERIA.
  const positive = result.classes ? result.classes[1] : "PNEUMONIA";
  const isPositive = result.prediction === positive;
  const badge = $("predBadge");
  badge.textContent = result.prediction;
  badge.className = `inline-flex min-h-7 items-center rounded-full border px-3 text-[11px] font-extrabold uppercase ${isPositive ? "border-red-200 bg-red-50 text-red-700" : "border-green-200 bg-green-50 text-green-700"}`;

  const truth = $("predTruth");
  if (trueLabel) {
    const correct = result.correct ?? (result.prediction === trueLabel);
    truth.textContent = `vrai: ${trueLabel} · ${correct ? "✓ correct" : "✗ erreur"}`;
    truth.className = `text-[11px] font-bold ${correct ? "text-green-600" : "text-red-600"}`;
  } else {
    truth.textContent = "image importée";
    truth.className = "text-[11px] text-slate-500";
  }

  const probCaption = $("predProbCaption");
  if (probCaption) probCaption.textContent = `Probabilité ${positive}`;
  $("predProbLabel").textContent = `${(prob * 100).toFixed(1)} %`;
  $("predProbBar").style.width = `${Math.min(100, prob * 100)}%`;
  $("predProbBar").className = `block h-full rounded-full transition-[width] duration-200 ${isPositive ? "bg-red-500" : "bg-green-500"}`;
  $("predThresholdMark").textContent = `seuil de décision = ${Number(result.threshold).toFixed(2)}`;
}

function bindTestPage() {
  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.addEventListener("click", () => setView(btn.dataset.view));
  }
  $("thresholdSlider").addEventListener("input", () => {
    $("thresholdValue").textContent = currentThreshold().toFixed(2);
  });
  $("reloadModelsBtn").addEventListener("click", loadModels);
  $("evalBtn").addEventListener("click", runEvaluate);
  $("predLoadSamples").addEventListener("click", loadPredSamples);
  $("testDataset").addEventListener("change", () => {
    updatePredClasses();
    loadPredSamples();
  });
  $("predUpload").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) predictUpload(file);
  });
  setView("train");
}

async function boot() {
  $("startBtn").addEventListener("click", startTraining);
  $("stopBtn").addEventListener("click", stopTraining);
  $("refreshBtn").addEventListener("click", async () => {
    await loadProject();
    await refreshStatus();
  });
  $("dataDir").addEventListener("change", loadSamples);

  bindTestPage();
  observeCharts();

  await loadProject();
  await refreshStatus();
  connectSocket();
}

boot().catch((error) => {
  console.error(error);
  $("logs").textContent = error.message;
});
