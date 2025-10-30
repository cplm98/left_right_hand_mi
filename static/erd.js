const fileSelect = document.getElementById("erd-file");
const datasetSelect = document.getElementById("erd-dataset");
const bandSelect = document.getElementById("erd-band");
const epochSlider = document.getElementById("epoch-slider");
const epochLabel = document.getElementById("epoch-label");
const prevButton = document.getElementById("prev-epoch");
const nextButton = document.getElementById("next-epoch");
const loadButton = document.getElementById("erd-load");
const statusText = document.getElementById("erd-status");
const plotEl = document.getElementById("erd-plot");
const summaryEl = document.getElementById("erd-summary");
const filterHpInput = document.getElementById("filter-hp");
const filterLpInput = document.getElementById("filter-lp");
const filterNotchBaseInput = document.getElementById("filter-notch-base");
const filterNotchHarmonicsInput = document.getElementById("filter-notch-harmonics");
const filterNotchQInput = document.getElementById("filter-notch-q");
const filterOrderInput = document.getElementById("filter-order");
const baselineSamplesInput = document.getElementById("baseline-samples");

const datasetLookup = new Map(
  (window.ERD_CONFIG?.datasets || []).map((item) => [item.value, item.label]),
);

const state = {
  epoch: 0,
  epochCount: 1,
  lastRequestId: 0,
  handledRequestId: 0,
};

function clampEpoch(value) {
  return Math.max(0, Math.min(value, state.epochCount - 1));
}

function parseNumericInput(input, { integer = false } = {}) {
  if (!input) {
    return undefined;
  }
  const raw = input.value?.trim();
  if (!raw) {
    return undefined;
  }
  const value = integer ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
  if (!Number.isFinite(value)) {
    console.warn("Ignoring non-numeric input", input.id, raw);
    return undefined;
  }
  return value;
}

function getFilterParams() {
  const params = {};
  const hp = parseNumericInput(filterHpInput);
  if (hp !== undefined) {
    params.hp = hp;
  }
  const lp = parseNumericInput(filterLpInput);
  if (lp !== undefined) {
    params.lp = lp;
  }
  const notchBase = parseNumericInput(filterNotchBaseInput);
  if (notchBase !== undefined) {
    params.notchBase = notchBase;
  }
  const notchHarmonics = parseNumericInput(filterNotchHarmonicsInput, {
    integer: true,
  });
  if (notchHarmonics !== undefined) {
    params.notchHarmonics = notchHarmonics;
  }
  const notchQ = parseNumericInput(filterNotchQInput);
  if (notchQ !== undefined) {
    params.notchQ = notchQ;
  }
  const order = parseNumericInput(filterOrderInput, { integer: true });
  if (order !== undefined) {
    params.order = order;
  }
  const baselineSamples = parseNumericInput(baselineSamplesInput, {
    integer: true,
  });
  if (baselineSamples !== undefined) {
    params.baselineSamples = baselineSamples;
  }
  return params;
}

function syncFilterInputs(filters) {
  if (!filters) {
    return;
  }
  if (filterHpInput) {
    filterHpInput.value =
      filters.hp === null || filters.hp === undefined ? "" : filters.hp;
  }
  if (filterLpInput) {
    filterLpInput.value =
      filters.lp === null || filters.lp === undefined ? "" : filters.lp;
  }
  if (filterNotchBaseInput) {
    filterNotchBaseInput.value =
      filters.notchBase === null || filters.notchBase === undefined
        ? ""
        : filters.notchBase;
  }
  if (filterNotchHarmonicsInput) {
    filterNotchHarmonicsInput.value =
      filters.notchHarmonics === null || filters.notchHarmonics === undefined
        ? ""
        : filters.notchHarmonics;
  }
  if (filterNotchQInput) {
    filterNotchQInput.value =
      filters.notchQ === null || filters.notchQ === undefined
        ? ""
        : filters.notchQ;
  }
  if (filterOrderInput) {
    filterOrderInput.value =
      filters.order === null || filters.order === undefined ? "" : filters.order;
  }
  if (baselineSamplesInput && filters.baselineSamples !== undefined) {
    baselineSamplesInput.value =
      filters.baselineSamples === null ? "" : filters.baselineSamples;
  }
}

function initializeFilterInputs() {
  const defaults = window.ERD_CONFIG?.filters || {};
  syncFilterInputs({
    hp: defaults.hp,
    lp: defaults.lp,
    notchBase: defaults.notchBase,
    notchHarmonics: defaults.notchHarmonics,
    notchQ: defaults.notchQ,
    order: defaults.order,
    baselineSamples: defaults.baselineSamples,
  });
}

function updateEpochControls(payload) {
  state.epochCount = payload.epochCount || 1;
  state.epoch = clampEpoch(payload.epoch ?? 0);

  epochSlider.min = "1";
  epochSlider.max = String(state.epochCount);
  epochSlider.value = String(state.epoch + 1);

  epochLabel.textContent = `Trial ${state.epoch + 1} / ${state.epochCount}`;
}

function formatSeconds(seconds) {
  return `${seconds.toFixed(2)} s`;
}

function renderSummary(payload) {
  if (!summaryEl) {
    return;
  }
  const datasetLabel = datasetLookup.get(payload.dataset) || payload.dataset;
  const bandLabel = payload.band?.replace("_band", "").toUpperCase();

  const windows = payload.windows || {};
  const baselineWindow = windows.baseline || {};
  const activeWindow = windows.active || {};

  const baselineStart = baselineWindow.startTime ?? 0;
  const baselineEnd = baselineWindow.endTime ?? baselineStart;
  const activeStart = activeWindow.startTime ?? 0;
  const activeEnd = activeWindow.endTime ?? activeStart;

  const baselineSamples = baselineWindow.samples ?? 0;
  const activeSamples = activeWindow.samples ?? 0;

  const baselineDuration =
    baselineSamples && payload.sampleRate
      ? `${baselineSamples} samples (${formatSeconds(
          baselineSamples / payload.sampleRate,
        )})`
      : "n/a";
  const activeDuration =
    activeSamples && payload.sampleRate
      ? `${activeSamples} samples (${formatSeconds(
          activeSamples / payload.sampleRate,
        )})`
      : "n/a";

  const filters = payload.filters || {};
  const filterParts = [
    filters.hp != null ? `HP ${filters.hp} Hz` : "HP off",
    filters.lp != null ? `LP ${filters.lp} Hz` : "LP off",
    filters.notchBase != null && filters.notchHarmonics
      ? `Notch ${filters.notchBase} Hz ×${filters.notchHarmonics}`
      : "Notch off",
    filters.notchBase != null && filters.notchHarmonics
      ? `Q ${filters.notchQ ?? "auto"}`
      : "",
    filters.order != null ? `Order ${filters.order}` : "",
  ].filter(Boolean);

  summaryEl.innerHTML = `
    <p><strong>File:</strong> ${payload.file} · <strong>Condition:</strong> ${datasetLabel} · <strong>Band:</strong> ${bandLabel}</p>
    <p><strong>Baseline:</strong> ${formatSeconds(baselineStart)} → ${formatSeconds(
      baselineEnd,
    )} (${baselineDuration}) · <strong>Active:</strong> ${formatSeconds(
      activeStart,
    )} → ${formatSeconds(activeEnd)} (${activeDuration}) · <strong>Samples per trial:</strong> ${
    payload.samplesPerTrial
  } @ ${payload.sampleRate} Hz</p>
    <p><strong>Filters:</strong> ${filterParts.join(" · ")}</p>
  `;
}

async function renderPlot(payload) {
  if (!plotEl) {
    return;
  }

  const showMessage = (message, tone = "info") => {
    Plotly.purge(plotEl);
    plotEl.innerHTML = `<p style="text-align:center;margin:3rem 0;color:${
      tone === "error" ? "#c0392b" : "#555"
    };">${message}</p>`;
  };

  const topomap = payload.topomap;
  const positions = payload.positions || [];
  const erdRange = payload.erdRange || [-1, 1];
  const finitePositions = positions.filter((point) =>
    Number.isFinite(point.erd),
  );
  console.log(
    "renderPlot",
    payload.dataset,
    payload.band,
    `epoch ${payload.epoch + 1}`,
    { hasTopomap: !!topomap, finitePositions: finitePositions.length },
  );

  const hasFiniteCell =
    topomap &&
    Array.isArray(topomap.z) &&
    topomap.z.some((row) => Array.isArray(row) && row.some(Number.isFinite));
  console.log(
    "renderPlot payload summary",
    {
      topomapPresent: Boolean(topomap),
      topomapRows: topomap?.z?.length ?? 0,
      topomapCols: topomap?.z?.[0]?.length ?? 0,
      finiteCells: topomap
        ? topomap.z.flat().filter(Number.isFinite).length
        : 0,
      finitePositions: finitePositions.length,
    },
  );

  if (!topomap || !Array.isArray(topomap.z) || !hasFiniteCell) {
    if (finitePositions.length === 0) {
      showMessage("Topomap unavailable for this dataset.");
      console.warn("ERD topomap unavailable; insufficient data");
      return;
    }

    const fallbackScatter = {
      type: "scatter",
      mode: "markers+text",
      x: finitePositions.map((point) => point.x),
      y: finitePositions.map((point) => point.y),
      marker: {
        size: 12,
        color: finitePositions.map((point) => point.erd),
        colorscale: "RdBu",
        cmin: erdRange[0],
        cmax: erdRange[1],
        cmid: 0,
        line: { color: "rgba(30, 41, 59, 0.55)", width: 1 },
      },
      text: finitePositions.map((point) => ` ${point.index}`),
      textposition: "middle center",
    };

    const radius = 1.05;
    const layout = {
      title: `ERD Channels · Trial ${state.epoch + 1}`,
      xaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        range: [-radius, radius],
        scaleanchor: "y",
        scaleratio: 1,
      },
      yaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        range: [-radius, radius],
        scaleratio: 1,
      },
      margin: { t: 80, r: 60, b: 60, l: 60 },
    };

    Plotly.purge(plotEl);
    await Plotly.newPlot(plotEl, [fallbackScatter], layout, {
      responsive: true,
      displaylogo: false,
    });
    Plotly.Plots.resize(plotEl);
    console.warn("Rendering fallback scatter only.", {
      traces: plotEl.data?.length,
    });
    return;
  }

  const channelLookup = new Map(
    (payload.channels || []).map((channel) => [channel.index, channel]),
  );

  const scatterHover = positions.map((position) => {
    const channel = channelLookup.get(position.index) || {};
    const erd = Number.isFinite(position.erd) ? position.erd : null;
    const baselineVal = channel.baseline?.[payload.band];
    const activeVal = channel.active?.[payload.band];
    const baselineText = Number.isFinite(baselineVal)
      ? Number(baselineVal).toFixed(4)
      : "n/a";
    const activeText = Number.isFinite(activeVal)
      ? Number(activeVal).toFixed(4)
      : "n/a";
    const erdText = erd === null ? "n/a" : erd.toFixed(4);
    return `Ch ${position.index}<br>ERD: ${erdText}<br>Baseline: ${baselineText}<br>Active: ${activeText}`;
  });

  const heatmapTrace = {
    type: "heatmap",
    x: topomap.x,
    y: topomap.y,
    z: topomap.z,
    colorscale: "RdBu",
    zmin: erdRange[0],
    zmax: erdRange[1],
    zmid: 0,
    colorbar: {
      title: "ERD (during − before) / before",
      titleside: "right",
    },
    hoverinfo: "skip",
  };

  const scatterTrace = {
    type: "scatter",
    mode: "markers+text",
    x: positions.map((point) => point.x),
    y: positions.map((point) => point.y),
    marker: {
      size: 10,
      color: positions.map((point) =>
        Number.isFinite(point.erd) ? point.erd : null,
      ),
      colorscale: "RdBu",
      cmin: erdRange[0],
      cmax: erdRange[1],
      cmid: 0,
      showscale: false,
      line: { color: "rgba(30, 41, 59, 0.55)", width: 1 },
    },
    hoverinfo: "text",
    hovertext: scatterHover,
    text: positions.map((point) => ` ${point.index}`),
    textposition: "middle center",
    textfont: {
      color: "rgba(17, 24, 39, 0.85)",
      size: 9,
    },
  };

  const radius = topomap.radius ?? 1;
  const margin = 80;

  const layout = {
    title: `ERD Topomap · Trial ${state.epoch + 1}`,
    xaxis: {
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      range: [-radius, radius],
      scaleanchor: "y",
      scaleratio: 1,
    },
    yaxis: {
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      range: [-radius, radius],
      scaleratio: 1,
    },
    shapes: [
      {
        type: "circle",
        xref: "x",
        yref: "y",
        x0: -radius,
        y0: -radius,
        x1: radius,
        y1: radius,
        line: { color: "rgba(30, 41, 59, 0.25)", width: 1.5 },
      },
    ],
    margin: { t: margin, r: margin * 0.6, b: margin * 0.6, l: margin * 0.6 },
  };

  const config = {
    responsive: true,
    displaylogo: false,
  };

  plotEl.innerHTML = "";
  try {
    Plotly.purge(plotEl);
    await Plotly.newPlot(plotEl, [heatmapTrace, scatterTrace], layout, config);
    console.log("renderPlot completed heatmap render", {
      traces: plotEl.data?.length,
      layout: plotEl.layout,
    });
    Plotly.Plots.resize(plotEl);
  } catch (error) {
    console.error("Failed to render ERD topomap", error);
    showMessage("Failed to render ERD topomap.", "error");
  }
}

async function fetchErd(epochOverride) {
  const file = fileSelect?.value;
  const dataset = datasetSelect?.value;
  const band = bandSelect?.value;

  if (!file || !dataset) {
    statusText.textContent = "Select a file and condition.";
    return;
  }

  const epoch = clampEpoch(
    typeof epochOverride === "number" ? epochOverride : state.epoch,
  );
  state.epoch = epoch;

  const requestId = ++state.lastRequestId;
  statusText.textContent = `Loading trial ${epoch + 1} …`;
  try {
    const params = new URLSearchParams({
      file,
      dataset,
      band,
      epoch: String(epoch),
    });
    const filterParams = getFilterParams();
    Object.entries(filterParams).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") {
        params.set(key, String(value));
      }
    });
    const response = await fetch(`/erd/data?${params.toString()}`);
    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || "Failed to load ERD data.");
    }
    const payload = await response.json();
    if (requestId < state.handledRequestId) {
      console.warn("Discarding stale ERD payload", { requestId, handled: state.handledRequestId });
      return;
    }
    state.handledRequestId = requestId;
    console.log("ERD payload", payload);
    syncFilterInputs({
      hp: payload.filters?.hp ?? null,
      lp: payload.filters?.lp ?? null,
      notchBase: payload.filters?.notchBase ?? null,
      notchHarmonics: payload.filters?.notchHarmonics ?? null,
      notchQ: payload.filters?.notchQ ?? null,
      order: payload.filters?.order ?? null,
      baselineSamples: payload.baseline?.samples ?? null,
    });
    updateEpochControls(payload);
    renderSummary(payload);
    await renderPlot(payload);
    statusText.textContent = `Showing trial ${state.epoch + 1} of ${state.epochCount}.`;
  } catch (error) {
    console.error(error);
    statusText.textContent = error.message;
    if (plotEl) {
      Plotly.purge(plotEl);
      plotEl.innerHTML =
        "<p style='text-align:center;margin:3rem 0;color:#c0392b;'>Failed to load ERD data.</p>";
    }
  }
}

function resetEpochAndFetch() {
  state.epoch = 0;
  fetchErd(0);
}

if (loadButton) {
  loadButton.addEventListener("click", () => resetEpochAndFetch());
}

if (fileSelect) {
  fileSelect.addEventListener("change", () => resetEpochAndFetch());
}

if (datasetSelect) {
  datasetSelect.addEventListener("change", () => resetEpochAndFetch());
}

if (bandSelect) {
  bandSelect.addEventListener("change", () => fetchErd());
}

if (epochSlider) {
  epochSlider.addEventListener("input", (event) => {
    const value = Number(event.target.value) || 1;
    epochLabel.textContent = `Trial ${value} / ${state.epochCount}`;
  });
  epochSlider.addEventListener("change", (event) => {
    const value = Number(event.target.value) || 1;
    state.epoch = clampEpoch(value - 1);
    fetchErd();
  });
}

if (prevButton) {
  prevButton.addEventListener("click", () => {
    if (state.epoch <= 0) {
      return;
    }
    state.epoch -= 1;
    epochSlider.value = String(state.epoch + 1);
    fetchErd();
  });
}

if (nextButton) {
  nextButton.addEventListener("click", () => {
    if (state.epoch >= state.epochCount - 1) {
      return;
    }
    state.epoch += 1;
    epochSlider.value = String(state.epoch + 1);
    fetchErd();
  });
}

// Initial load
initializeFilterInputs();
fetchErd();
