const $ = (id) => document.getElementById(id);

const fileInput = $("file");
const drop = $("drop");
const recBtn = $("rec");
const statusEl = $("status");
const player = $("player");
const out = $("out");
const meta = $("meta");
const copyBtn = $("copy");
const healthEl = $("health");
const levelCanvas = $("level");
const levelCtx = levelCanvas.getContext("2d");

let recCtx = null;
let recChunks = [];
let recNode = null;
let recSource = null;
let recStream = null;
let analyser = null;
let levelRaf = 0;
let recording = false;
let lastText = "";

function setStatus(text) {
  statusEl.textContent = text;
}

function encodeWav(float32, sampleRate) {
  const n = float32.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  writeStr(36, "data");
  v.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

function drawIdleLevel() {
  const w = levelCanvas.width;
  const h = levelCanvas.height;
  levelCtx.clearRect(0, 0, w, h);
  levelCtx.beginPath();
  levelCtx.arc(w / 2, h / 2, 78, 0, Math.PI * 2);
  levelCtx.strokeStyle = "rgba(13, 138, 124, 0.18)";
  levelCtx.lineWidth = 3;
  levelCtx.stroke();
}

function drawLevelFrame() {
  if (!analyser) return;
  const w = levelCanvas.width;
  const h = levelCanvas.height;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    const v = (data[i] - 128) / 128;
    sum += v * v;
  }
  const rms = Math.sqrt(sum / data.length);
  const radius = 78 + Math.min(42, rms * 110);

  levelCtx.clearRect(0, 0, w, h);
  levelCtx.beginPath();
  levelCtx.arc(w / 2, h / 2, radius, 0, Math.PI * 2);
  levelCtx.strokeStyle = "rgba(194, 59, 59, 0.45)";
  levelCtx.lineWidth = 4;
  levelCtx.stroke();

  levelCtx.beginPath();
  const mid = h / 2;
  const left = 40;
  const right = w - 40;
  levelCtx.moveTo(left, mid);
  for (let i = 0; i < data.length; i++) {
    const x = left + ((right - left) * i) / data.length;
    const y = mid + ((data[i] - 128) / 128) * 36;
    levelCtx.lineTo(x, y);
  }
  levelCtx.strokeStyle = "rgba(15, 42, 50, 0.35)";
  levelCtx.lineWidth = 1.5;
  levelCtx.stroke();

  levelRaf = requestAnimationFrame(drawLevelFrame);
}

function stopLevel() {
  cancelAnimationFrame(levelRaf);
  analyser = null;
  drawIdleLevel();
}

function setTranscript(html, { empty = false, reject = false, text = "" } = {}) {
  out.innerHTML = html;
  out.dataset.empty = empty ? "true" : "false";
  out.classList.toggle("reject", reject);
  out.classList.remove("fresh");
  void out.offsetWidth;
  out.classList.add("fresh");
  lastText = text;
  copyBtn.disabled = !text;
}

async function sendBlob(blob, name) {
  setStatus("Đang chuyển thành chữ…");
  const url = URL.createObjectURL(blob);
  player.src = url;
  player.classList.add("show");

  const fd = new FormData();
  fd.append("file", blob, name);
  let res;
  let data = {};
  try {
    res = await fetch("/api/predict", { method: "POST", body: fd });
    data = await res.json().catch(() => ({}));
  } catch (err) {
    setTranscript("Không kết nối được máy chủ.", { reject: true });
    setStatus("Lỗi kết nối");
    meta.hidden = true;
    return;
  }

  if (!res.ok) {
    const msg = data.detail || "Lỗi nhận dạng";
    setTranscript(msg, { reject: true });
    setStatus("Lỗi");
    meta.hidden = true;
    return;
  }

  renderResult(data);
  setStatus("Xong");
}

function renderResult(data) {
  if (data.rejected) {
    setTranscript("Không nhận diện được đoạn này. Thử nói rõ hơn, nghỉ giữa các từ, hoặc thu lại.", {
      reject: true,
    });
    meta.hidden = false;
    meta.textContent = `Đoạn con: ${data.n_segments ?? 0}`;
    return;
  }

  const text = data.text || data.word_text || "";
  setTranscript(text, { text });
  meta.hidden = false;
  meta.textContent = [
    data.n_words != null ? `${data.n_words} từ` : null,
    data.n_segments != null ? `${data.n_segments} đoạn` : null,
    data.confidence != null ? `Tin cậy TB ${Math.round(data.confidence * 100)}%` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

async function startRec() {
  if (recording) return;
  try {
    recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    setStatus("Không mở được mic — kiểm tra quyền trình duyệt");
    return;
  }

  recCtx = new AudioContext({ sampleRate: 16000 });
  recSource = recCtx.createMediaStreamSource(recStream);
  analyser = recCtx.createAnalyser();
  analyser.fftSize = 2048;
  recNode = recCtx.createScriptProcessor(4096, 1, 1);
  recChunks = [];
  recNode.onaudioprocess = (e) => {
    recChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  recSource.connect(analyser);
  analyser.connect(recNode);
  const mute = recCtx.createGain();
  mute.gain.value = 0;
  recNode.connect(mute);
  mute.connect(recCtx.destination);

  recording = true;
  recBtn.classList.add("recording");
  recBtn.setAttribute("aria-pressed", "true");
  setStatus("Đang nghe… nói đoạn, nghỉ ngắn giữa các từ, rồi bấm mic để dừng");
  drawLevelFrame();
}

async function stopRec() {
  if (!recording) return;
  recording = false;
  recBtn.classList.remove("recording");
  recBtn.setAttribute("aria-pressed", "false");
  stopLevel();

  const sr = recCtx.sampleRate || 16000;
  try {
    recNode.disconnect();
    recSource.disconnect();
  } catch (_) {}
  recStream.getTracks().forEach((t) => t.stop());
  await recCtx.close();

  const len = recChunks.reduce((a, c) => a + c.length, 0);
  if (len < sr * 0.15) {
    setStatus("Đoạn quá ngắn — thu lại");
    return;
  }
  const samples = new Float32Array(len);
  let o = 0;
  recChunks.forEach((c) => {
    samples.set(c, o);
    o += c.length;
  });
  const blob = encodeWav(samples, sr);
  await sendBlob(blob, "mic.wav");
}

recBtn.addEventListener("click", () => {
  if (recording) stopRec();
  else startRec();
});

function handleFiles(files) {
  const f = files && files[0];
  if (f) sendBlob(f, f.name);
}

fileInput.addEventListener("change", () => handleFiles(fileInput.files));

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

["dragenter", "dragover"].forEach((ev) => {
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((ev) => {
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
  });
});
drop.addEventListener("drop", (e) => {
  handleFiles(e.dataTransfer.files);
});

copyBtn.addEventListener("click", async () => {
  if (!lastText) return;
  try {
    await navigator.clipboard.writeText(lastText);
    copyBtn.textContent = "Đã chép";
    setTimeout(() => {
      copyBtn.textContent = "Sao chép";
    }, 1200);
  } catch {
    copyBtn.textContent = "Không chép được";
  }
});

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    if (h.model) {
      healthEl.textContent = "Model sẵn sàng";
      healthEl.className = "badge badge-ok";
    } else {
      healthEl.textContent = "Chưa train model";
      healthEl.className = "badge badge-off";
      healthEl.title = "Chạy extract_features.py rồi train.py";
    }
  })
  .catch(() => {
    healthEl.textContent = "Máy chủ lỗi";
    healthEl.className = "badge badge-off";
  });

drawIdleLevel();
