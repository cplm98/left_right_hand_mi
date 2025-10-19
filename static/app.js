const select = document.getElementById("file-select");
const segmentSelect = document.getElementById("segment-select");
const loadButton = document.getElementById("load-button");
const statusText = document.getElementById("status");
const plotContainer = document.getElementById("plot");

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

function renderPlot(payload) {
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
    margin: { t: 50, r: 20, b: 50, l: 60 },
    legend: { orientation: "h", x: 0, y: 1.1 },
  };

  Plotly.react(plotContainer, traces, layout, {
    responsive: true,
    displaylogo: false,
  });
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
    renderPlot(payload);
    const segmentLabel = payload.segment ?? "segment";
    const rateText = payload.sampleRate ? ` @ ${payload.sampleRate} Hz` : "";
    statusText.textContent = `Showing ${filename} • ${segmentLabel} (${payload.time.length} samples${rateText}).`;
  } catch (error) {
    console.error(error);
    statusText.textContent = error.message;
    Plotly.purge(plotContainer);
  }
}

if (loadButton) {
  loadButton.addEventListener("click", loadSelectedFile);
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
