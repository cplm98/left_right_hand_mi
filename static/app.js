const select = document.getElementById("file-select");
const segmentSelect = document.getElementById("segment-select");
const loadButton = document.getElementById("load-button");
const statusText = document.getElementById("status");
const plotSection = document.getElementById("plot");
const plotArea = document.getElementById("plot-area");
const fullscreenButton = document.getElementById("fullscreen");
const legendContainer = document.getElementById("channel-legend");
const legendList = document.getElementById("legend-list");
const selectAllButton = document.getElementById("select-all");
const deselectAllButton = document.getElementById("deselect-all");

let traceVisibility = [];

async function fetchChannels(filename, segment) {
  const params = new URLSearchParams({ file: filename });
  if (segment) {
    params.append("segment", segment);
  }
  const response = await fetch(`/data?${params.toString()}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Failed to load ${filename}`);
  }
  return response.json();
}

function updateSegments(segments, activeSegment) {
  if (!segmentSelect) {
    return;
  }

  segmentSelect.innerHTML = "";
  if (!segments || segments.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No segments found";
    segmentSelect.appendChild(option);
    segmentSelect.disabled = true;
    return;
  }

  segmentSelect.disabled = false;
  segments.forEach((segment) => {
    const option = document.createElement("option");
    option.value = segment;
    option.textContent = segment;
    segmentSelect.appendChild(option);
  });
  segmentSelect.value =
    activeSegment && segments.includes(activeSegment)
      ? activeSegment
      : segments[0];
}

function buildLegend(payload) {
  if (!legendContainer || !legendList) {
    return;
  }

  legendList.innerHTML = "";
  traceVisibility = payload.channels.map(() => true);

  payload.channels.forEach((channel, idx) => {
    const item = document.createElement("label");
    item.className = "legend-item";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.dataset.traceIndex = String(idx);

    const span = document.createElement("span");
    span.textContent = channel.name;

    checkbox.addEventListener("change", (event) => {
      const target = event.target;
      const traceIndex = Number(target.dataset.traceIndex);
      const visible = target.checked;
      traceVisibility[traceIndex] = visible;
      if (plotArea && plotArea.data) {
        Plotly.restyle(plotArea, { visible: visible ? true : false }, [
          traceIndex,
        ]);
      }
    });

    item.appendChild(checkbox);
    item.appendChild(span);
    legendList.appendChild(item);
  });
}

async function renderPlot(payload) {
  if (!plotArea) {
    return;
  }

  const xLabel = payload.sampleRate
    ? `Time (s) @ ${payload.sampleRate} Hz`
    : "Sample index";
  const timeAxis =
    payload.sampleRate && Array.isArray(payload.time)
      ? payload.time.map((x) => x / payload.sampleRate)
      : payload.time;

  const traces = payload.channels.map((channel) => ({
    x: timeAxis,
    y: channel.values,
    mode: "lines",
    name: channel.name,
  }));

  const layout = {
    title: `${payload.segment} · ${payload.file}`,
    xaxis: { title: xLabel },
    yaxis: { title: "Amplitude" },
    margin: { t: 50, r: 20, b: 50, l: 50 },
    showlegend: false,
    autosize: true,
  };

  await Plotly.react(plotArea, traces, layout, {
    responsive: true,
    displaylogo: false,
  });

  buildLegend(payload);
  Plotly.Plots.resize(plotArea);
}

async function loadSelectedFile() {
  const filename = select?.value;
  if (!filename) {
    statusText.textContent = "No file selected.";
    return;
  }

  const segment =
    segmentSelect && !segmentSelect.disabled ? segmentSelect.value : undefined;

  statusText.textContent = `Loading ${filename}${
    segment ? ` (${segment})` : ""
  } ...`;
  try {
    const payload = await fetchChannels(filename, segment);
    updateSegments(payload.segments, payload.segment);
    await renderPlot(payload);
    const segmentLabel = payload.segment ?? "segment";
    const rateText = payload.sampleRate ? ` @ ${payload.sampleRate} Hz` : "";
    const channelSummary =
      payload.channelCount === payload.availableChannels ||
      payload.availableChannels == null
        ? `${payload.channelCount} channels`
        : `${payload.channelCount}/${payload.availableChannels} channels`;
    statusText.textContent = `Showing ${filename} • ${segmentLabel} (${channelSummary}, ${payload.time.length} samples${rateText}).`;
  } catch (error) {
    console.error(error);
    statusText.textContent = error.message;
    if (plotArea) {
      Plotly.purge(plotArea);
    }
  }
}

if (loadButton) {
  loadButton.addEventListener("click", loadSelectedFile);
}

function setAllTraces(visible) {
  if (!legendList) {
    return;
  }
  const checkboxes = legendList.querySelectorAll("input[type='checkbox']");
  checkboxes.forEach((checkbox) => {
    const traceIndex = Number(checkbox.dataset.traceIndex);
    if (traceVisibility[traceIndex] === visible) {
      return;
    }
    checkbox.checked = visible;
    traceVisibility[traceIndex] = visible;
  });
  if (plotArea && plotArea.data) {
    const flag = visible ? true : false;
    const indices = traceVisibility.map((_, idx) => idx);
    Plotly.restyle(plotArea, { visible: flag }, indices);
  }
}

if (selectAllButton) {
  selectAllButton.addEventListener("click", () => setAllTraces(true));
}

if (deselectAllButton) {
  deselectAllButton.addEventListener("click", () => setAllTraces(false));
}

if (select) {
  select.addEventListener("change", () => loadSelectedFile());
}

if (segmentSelect) {
  segmentSelect.addEventListener("change", () => loadSelectedFile());
}

if (select && select.options.length > 0) {
  loadSelectedFile();
}

function toggleFullscreen() {
  if (!plotSection) {
    return;
  }
  const fullscreenElement = document.fullscreenElement;
  if (!fullscreenElement) {
    plotSection.requestFullscreen?.().catch((error) => {
      console.error("Failed to enter fullscreen:", error);
    });
  } else {
    document.exitFullscreen?.().catch((error) => {
      console.error("Failed to exit fullscreen:", error);
    });
  }
}

if (fullscreenButton) {
  fullscreenButton.addEventListener("click", toggleFullscreen);
}

document.addEventListener("fullscreenchange", () => {
  if (!fullscreenButton) {
    return;
  }
  if (document.fullscreenElement) {
    fullscreenButton.textContent = "Exit fullscreen";
    if (plotArea) {
      Plotly.Plots.resize(plotArea);
    }
  } else {
    fullscreenButton.textContent = "Fullscreen";
    if (plotArea) {
      Plotly.Plots.resize(plotArea);
    }
  }
});

window.addEventListener("resize", () => {
  if (plotArea) {
    Plotly.Plots.resize(plotArea);
  }
});
