/* Single-page upload + result rendering for the fall-detection web demo. */

(function () {
  "use strict";

  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("video-input");
  const fileLabel = fileInput.parentElement;
  const fileLabelText = fileLabel.querySelector("span");
  const runBtn = document.getElementById("run-btn");
  const progressBox = document.getElementById("progress");
  const progressText = document.getElementById("progress-text");
  const errorBox = document.getElementById("error");
  const resultsBox = document.getElementById("results");
  const annotatedVideo = document.getElementById("annotated-video");
  const summaryList = document.getElementById("summary-list");
  const narrationBox = document.getElementById("narration");
  const eventsBody = document.querySelector("#events-table tbody");
  const noEventsMsg = document.getElementById("no-events");
  const downloadList = document.getElementById("download-list");
  const probChartCanvas = document.getElementById("prob-chart");

  let probChart = null;

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
      fileLabel.classList.add("has-file");
      fileLabelText.textContent = `${file.name} — ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
    } else {
      fileLabel.classList.remove("has-file");
      fileLabelText.textContent = "Click or drop a clip (.mp4, .mov, .avi, .mkv, .webm — up to 200 MB)";
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!fileInput.files[0]) return;

    errorBox.classList.add("hidden");
    resultsBox.classList.add("hidden");
    progressBox.classList.remove("hidden");
    progressText.textContent = "Uploading and processing — this may take a minute for longer clips…";
    runBtn.disabled = true;

    const formData = new FormData(form);

    try {
      const response = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(detail.detail || `HTTP ${response.status}`);
      }
      const payload = await response.json();
      renderResults(payload);
    } catch (err) {
      errorBox.textContent = `Error: ${err.message}`;
      errorBox.classList.remove("hidden");
    } finally {
      progressBox.classList.add("hidden");
      runBtn.disabled = false;
    }
  });

  function renderResults(payload) {
    annotatedVideo.src = payload.annotated_url;
    annotatedVideo.load();

    summaryList.innerHTML = "";
    addKv("Frames processed", payload.processed_frames);
    addKv("Source FPS", payload.fps.toFixed(2));
    addKv("Clip duration", `${payload.duration_sec.toFixed(2)} s`);
    addKv("Wall-clock time", `${payload.elapsed_sec.toFixed(1)} s`);
    addKv("Confirmed falls", payload.events.length);
    addKv("Alarm threshold", payload.threshold_used.toFixed(2));
    addKv("Models", payload.ensemble_models.join(" + "));

    if (payload.narration) {
      narrationBox.textContent = payload.narration;
      narrationBox.classList.remove("empty");
    } else {
      narrationBox.textContent = "No narration was generated. Toggle the Gemini option (and set GOOGLE_API_KEY) to enable.";
      narrationBox.classList.add("empty");
    }

    eventsBody.innerHTML = "";
    if (payload.events.length === 0) {
      noEventsMsg.classList.remove("hidden");
    } else {
      noEventsMsg.classList.add("hidden");
      payload.events.forEach((ev, i) => {
        const tr = document.createElement("tr");
        tr.classList.add("alarm-row");
        tr.innerHTML = `<td>${i + 1}</td><td>${ev.frame}</td><td>${ev.time_sec.toFixed(2)}</td><td>${ev.confidence.toFixed(3)}</td>`;
        eventsBody.appendChild(tr);
      });
    }

    downloadList.innerHTML = "";
    addDownload("Annotated MP4", payload.annotated_url);
    addDownload("Timeline JSON", payload.timeline_url);

    drawProbChart(payload);

    resultsBox.classList.remove("hidden");
    resultsBox.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function addKv(label, value) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="k">${label}</span><span class="v">${value}</span>`;
    summaryList.appendChild(li);
  }

  function addDownload(label, href) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = href;
    a.textContent = label;
    a.setAttribute("download", "");
    li.appendChild(a);
    downloadList.appendChild(li);
  }

  function drawProbChart(payload) {
    const labels = payload.probabilities.map((p) => p.time_sec);
    const data = payload.probabilities.map((p) => p.p_fall);
    const threshold = payload.threshold_used;

    const eventLines = payload.events.map((ev) => ev.time_sec);

    if (probChart) {
      probChart.destroy();
    }

    probChart = new Chart(probChartCanvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "P(fall)",
            data,
            borderColor: "#4f8cff",
            backgroundColor: "rgba(79,140,255,0.18)",
            fill: true,
            pointRadius: 0,
            borderWidth: 1.5,
            tension: 0.2,
          },
          {
            label: `Threshold (${threshold.toFixed(2)})`,
            data: labels.map(() => threshold),
            borderColor: "#ffb454",
            borderDash: [6, 4],
            pointRadius: 0,
            borderWidth: 1,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        animation: false,
        scales: {
          x: {
            ticks: { color: "#8b95a7", maxTicksLimit: 12, callback: (v, i) => labels[i].toFixed(1) },
            grid: { color: "rgba(255,255,255,0.04)" },
            title: { display: true, text: "Time (s)", color: "#8b95a7" },
          },
          y: {
            min: 0,
            max: 1,
            ticks: { color: "#8b95a7" },
            grid: { color: "rgba(255,255,255,0.04)" },
            title: { display: true, text: "P(fall)", color: "#8b95a7" },
          },
        },
        plugins: {
          legend: { labels: { color: "#e8eaed" } },
          tooltip: { mode: "index", intersect: false },
        },
      },
      plugins: [
        {
          id: "eventMarkers",
          afterDatasetsDraw(chart) {
            const { ctx, chartArea, scales } = chart;
            ctx.save();
            ctx.strokeStyle = "rgba(255,89,97,0.7)";
            ctx.lineWidth = 1.2;
            ctx.setLineDash([3, 3]);
            eventLines.forEach((t) => {
              const x = scales.x.getPixelForValue(t);
              if (x >= chartArea.left && x <= chartArea.right) {
                ctx.beginPath();
                ctx.moveTo(x, chartArea.top);
                ctx.lineTo(x, chartArea.bottom);
                ctx.stroke();
              }
            });
            ctx.restore();
          },
        },
      ],
    });
  }
})();
