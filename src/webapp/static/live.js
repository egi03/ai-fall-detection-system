/* Live webcam mode for the fall-detection web demo.
 *
 * Flow:
 *   1. getUserMedia() → <video>
 *   2. WebSocket /ws/live opens; server sends {type: "config"} with skeleton
 *      topology + threshold + model list.
 *   3. A capture loop reads <video> into an offscreen <canvas>, encodes JPEG,
 *      and sends the bytes over the socket at ~12 fps (matched to TARGET_FPS).
 *   4. Each server reply paints the overlay canvas, updates the HUD, and pushes
 *      a sample into the rolling Chart.js line.
 *
 * No frames are ever stored; pixels stay in volatile memory only as long as the
 * loop is running.
 */

(function () {
  "use strict";

  const TARGET_FPS = 12;          // capture cadence; server is happy at this rate
  const JPEG_QUALITY = 0.6;       // good enough for pose, ~30 KB/frame
  const CAPTURE_WIDTH = 480;      // downscale before encoding to save bandwidth
  const CHART_WINDOW_SECONDS = 30;

  /* ------------------------------------------------------------------ */
  /* Tab switching                                                       */
  /* ------------------------------------------------------------------ */
  document.addEventListener("DOMContentLoaded", () => {
    const tabs = document.querySelectorAll(".tab");
    const views = {
      upload: document.getElementById("upload-view"),
      live: document.getElementById("live-view"),
    };
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const target = tab.dataset.tab;
        tabs.forEach((t) => {
          const active = t === tab;
          t.classList.toggle("active", active);
          t.setAttribute("aria-selected", active ? "true" : "false");
        });
        Object.entries(views).forEach(([name, el]) => {
          if (!el) return;
          el.classList.toggle("hidden", name !== target);
        });
      });
    });

    setupLive();
  });

  /* ------------------------------------------------------------------ */
  /* Live mode                                                          */
  /* ------------------------------------------------------------------ */
  function setupLive() {
    const startBtn = document.getElementById("live-start");
    const stopBtn = document.getElementById("live-stop");
    const statusEl = document.getElementById("live-status");
    const errorEl = document.getElementById("live-error");
    const stage = document.getElementById("live-stage");
    const results = document.getElementById("live-results");
    const eventsCard = document.getElementById("live-events-card");
    const video = document.getElementById("live-video");
    const overlay = document.getElementById("live-overlay");
    const badge = document.getElementById("live-badge");
    const fpsLabel = document.getElementById("live-fps");
    const stateOut = document.getElementById("ls-state");
    const probOut = document.getElementById("ls-prob");
    const windowOut = document.getElementById("ls-window");
    const fallsOut = document.getElementById("ls-falls");
    const sendOut = document.getElementById("ls-send");
    const latencyOut = document.getElementById("ls-latency");
    const thresholdOut = document.getElementById("ls-threshold");
    const modelsOut = document.getElementById("ls-models");
    const eventsBody = document.querySelector("#live-events-table tbody");
    const chartCanvas = document.getElementById("live-chart");

    let stream = null;
    let socket = null;
    let captureTimer = null;
    let captureCanvas = null;
    let captureCtx = null;
    let cfg = null;                 // server-sent topology + threshold
    let chart = null;
    let chartData = [];             // [{t, p}]
    let frameCounter = 0;
    let serverFrames = 0;
    let lastFpsTick = performance.now();
    let recentSent = [];            // sliding window of send timestamps
    let inFlight = 0;
    let stopping = false;

    startBtn.addEventListener("click", start);
    stopBtn.addEventListener("click", stop);
    window.addEventListener("beforeunload", stop);

    async function start() {
      stopping = false;
      errorEl.classList.add("hidden");
      statusEl.textContent = "Requesting camera…";
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
      } catch (err) {
        showError(`Could not access webcam: ${err.message}`);
        statusEl.textContent = "Idle.";
        return;
      }
      video.srcObject = stream;
      await new Promise((res) => {
        if (video.readyState >= 2) res();
        else video.addEventListener("loadedmetadata", res, { once: true });
      });

      stage.classList.remove("hidden");
      results.classList.remove("hidden");
      startBtn.classList.add("hidden");
      stopBtn.classList.remove("hidden");

      sizeOverlayToVideo();
      window.addEventListener("resize", sizeOverlayToVideo);

      statusEl.textContent = "Connecting to server…";
      const proto = location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${proto}://${location.host}/ws/live`);
      socket.binaryType = "arraybuffer";

      socket.addEventListener("open", onSocketOpen);
      socket.addEventListener("message", onSocketMessage);
      socket.addEventListener("close", onSocketClose);
      socket.addEventListener("error", () => showError("WebSocket error"));
    }

    function onSocketOpen() {
      statusEl.textContent = "Connected. Waiting for model…";
      // Prepare offscreen capture canvas at downscaled size.
      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      const scale = Math.min(1, CAPTURE_WIDTH / vw);
      captureCanvas = document.createElement("canvas");
      captureCanvas.width = Math.round(vw * scale);
      captureCanvas.height = Math.round(vh * scale);
      captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });
    }

    function onSocketMessage(event) {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      if (payload.type === "config") {
        cfg = payload;
        thresholdOut.textContent = payload.threshold.toFixed(2);
        modelsOut.textContent = (payload.models || []).join(" + ") || "—";
        windowOut.textContent = `0 / ${payload.window_size}`;
        statusEl.textContent = "Streaming. Falls will be marked here.";
        initChart();
        startCaptureLoop();
        return;
      }
      if (payload.type === "error") {
        showError(payload.detail || "Server error");
        return;
      }
      if (payload.type === "pong") return;
      if (payload.type !== "frame") return;

      inFlight = Math.max(0, inFlight - 1);
      serverFrames += 1;
      renderFrame(payload);
    }

    function onSocketClose() {
      if (!stopping) {
        statusEl.textContent = "Disconnected.";
      }
      stop();
    }

    function startCaptureLoop() {
      if (captureTimer) clearInterval(captureTimer);
      const intervalMs = Math.round(1000 / TARGET_FPS);
      captureTimer = setInterval(captureAndSend, intervalMs);
    }

    function captureAndSend() {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      // Drop frames if the server is more than 3 behind — keeps latency bounded
      // when the client outruns inference (common on slower CPUs).
      if (inFlight >= 3) return;
      if (!video.videoWidth) return;

      captureCtx.drawImage(
        video,
        0,
        0,
        captureCanvas.width,
        captureCanvas.height,
      );
      captureCanvas.toBlob(
        (blob) => {
          if (!blob || !socket || socket.readyState !== WebSocket.OPEN) return;
          blob.arrayBuffer().then((buf) => {
            socket.send(buf);
            inFlight += 1;
            frameCounter += 1;
            const now = performance.now();
            recentSent.push(now);
            while (recentSent.length && recentSent[0] < now - 1000) {
              recentSent.shift();
            }
          });
        },
        "image/jpeg",
        JPEG_QUALITY,
      );
    }

    function renderFrame(payload) {
      drawOverlay(payload);
      updateBadge(payload.state);
      stateOut.textContent = payload.state;
      probOut.textContent = payload.p_fall.toFixed(3);
      const filled = Math.round(payload.window_progress * cfg.window_size);
      windowOut.textContent = `${filled} / ${cfg.window_size}`;
      fallsOut.textContent = payload.fall_count;
      latencyOut.textContent = `${payload.elapsed_ms.toFixed(1)} ms`;

      pushChart(payload.time_sec, payload.p_fall);

      if (payload.event) {
        appendEvent(payload.event);
      }

      const now = performance.now();
      if (now - lastFpsTick >= 500) {
        const sendFps = recentSent.length;
        sendOut.textContent = `${sendFps.toFixed(0)} fps`;
        fpsLabel.textContent = `${sendFps.toFixed(0)} fps`;
        lastFpsTick = now;
      }
    }

    function drawOverlay(payload) {
      const ctx = overlay.getContext("2d");
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      if (!payload.keypoints || !cfg) return;

      const w = overlay.width;
      const h = overlay.height;
      const color = stateColor(payload.state);
      const kps = payload.keypoints;
      const visMin = 0.5;

      // Bones
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      cfg.bones.forEach(([i, j]) => {
        const a = kps[i];
        const b = kps[j];
        if (!a || !b) return;
        if (a[2] < visMin || b[2] < visMin) return;
        ctx.beginPath();
        ctx.moveTo(a[0] * w, a[1] * h);
        ctx.lineTo(b[0] * w, b[1] * h);
        ctx.stroke();
      });

      // Joints
      ctx.fillStyle = color;
      kps.forEach((kp) => {
        if (kp[2] < visMin) return;
        ctx.beginPath();
        ctx.arc(kp[0] * w, kp[1] * h, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function stateColor(state) {
      switch (state) {
        case "FALL_CONFIRMED":
          return "#ff5961";
        case "IMPACT_DETECTED":
          return "#ffa040";
        case "IMMINENT":
          return "#ffd84a";
        case "RECOVERED":
          return "#43c78a";
        case "NO_PERSON":
        case "INIT":
          return "#8b95a7";
        default:
          return "#43c78a";
      }
    }

    function updateBadge(state) {
      badge.textContent = state.replace(/_/g, " ");
      badge.className = "state-badge " + ({
        FALL_CONFIRMED: "state-alarm",
        IMPACT_DETECTED: "state-warn",
        IMMINENT: "state-imminent",
        RECOVERED: "state-recovered",
        NO_PERSON: "state-muted",
        INIT: "state-muted",
      }[state] || "state-normal");
    }

    function appendEvent(event) {
      eventsCard.classList.remove("hidden");
      const row = document.createElement("tr");
      row.classList.add("alarm-row");
      const idx = eventsBody.children.length + 1;
      row.innerHTML = `<td>${idx}</td><td>${event.frame}</td><td>${event.time_sec.toFixed(2)}</td><td>${event.confidence.toFixed(3)}</td>`;
      eventsBody.appendChild(row);
    }

    function sizeOverlayToVideo() {
      const rect = video.getBoundingClientRect();
      // Match overlay drawing buffer to displayed video size; CSS already
      // overlaps them, but we need pixel-accurate coords here.
      overlay.width = Math.round(rect.width);
      overlay.height = Math.round(rect.height);
    }

    function initChart() {
      chartData = [];
      if (chart) chart.destroy();
      chart = new Chart(chartCanvas, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            {
              label: "P(fall)",
              data: [],
              borderColor: "#4f8cff",
              backgroundColor: "rgba(79,140,255,0.18)",
              fill: true,
              pointRadius: 0,
              borderWidth: 1.5,
              tension: 0.2,
            },
            {
              label: `Threshold (${cfg.threshold.toFixed(2)})`,
              data: [],
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
              ticks: { color: "#8b95a7", maxTicksLimit: 8 },
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
      });
    }

    function pushChart(t, p) {
      if (!chart) return;
      chartData.push({ t, p });
      const cutoff = t - CHART_WINDOW_SECONDS;
      while (chartData.length && chartData[0].t < cutoff) {
        chartData.shift();
      }
      chart.data.labels = chartData.map((d) => d.t.toFixed(1));
      chart.data.datasets[0].data = chartData.map((d) => d.p);
      chart.data.datasets[1].data = chartData.map(() => cfg.threshold);
      chart.update("none");
    }

    function stop() {
      stopping = true;
      if (captureTimer) {
        clearInterval(captureTimer);
        captureTimer = null;
      }
      if (socket) {
        try {
          if (socket.readyState === WebSocket.OPEN) socket.close();
        } catch (e) {
          /* noop */
        }
        socket = null;
      }
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
      }
      video.srcObject = null;
      startBtn.classList.remove("hidden");
      stopBtn.classList.add("hidden");
      window.removeEventListener("resize", sizeOverlayToVideo);
      if (statusEl.textContent === "Streaming. Falls will be marked here.") {
        statusEl.textContent = "Stopped.";
      }
    }

    function showError(msg) {
      errorEl.textContent = msg;
      errorEl.classList.remove("hidden");
    }
  }
})();
