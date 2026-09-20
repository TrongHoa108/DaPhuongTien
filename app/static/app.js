const $ = (id) => document.getElementById(id);
const fileInput = $("file");
const recBtn = $("rec");
const stopBtn = $("stop");
const statusEl = $("status");
const player = $("player");
const out = $("out");

let recCtx = null;
let recChunks = [];
let recNode = null;
let recSource = null;
let recStream = null;

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

async function sendBlob(blob, name) {
  statusEl.textContent = "Đang nhận dạng…";
  player.src = URL.createObjectURL(blob);
  const fd = new FormData();
  fd.append("file", blob, name);
  const res = await fetch("/api/predict", { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    out.innerHTML = `<h2>2. Kết quả</h2><p class="reject">${data.detail || "Lỗi nhận dạng"}</p>`;
    statusEl.textContent = "Lỗi";
    return;
  }
  renderResult(data);
  statusEl.textContent = "Xong";
}

function renderResult(data) {
  if (data.rejected) {
    out.innerHTML = `
      <h2>2. Kết quả</h2>
      <p class="reject">Không chắc — có thể nhiễu hoặc từ ngoài danh sách.</p>
      <p class="meta">Khoảng cách trung bình: ${data.mean_distance}</p>
    `;
    return;
  }
  const nb = (data.neighbors || [])
    .map((n) => `<li><span>${n.word_text} <span class="meta">${n.word_code}</span></span><span>${n.distance}</span></li>`)
    .join("");
  const f = data.features || {};
  out.innerHTML = `
    <h2>2. Kết quả</h2>
    <div class="word">${data.word_text}</div>
    <p class="meta">${data.word_code} · tin cậy ${Math.round((data.confidence || 0) * 100)}%</p>
    <p class="meta">Năng lượng ${f.energy_mean} · ZCR ${f.zcr_mean} · Pitch ${f.pitch_mean} Hz</p>
    <ul class="neighbors">${nb}</ul>
  `;
}

fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  if (f) sendBlob(f, f.name);
});

recBtn.addEventListener("click", async () => {
  recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recCtx = new AudioContext({ sampleRate: 16000 });
  recSource = recCtx.createMediaStreamSource(recStream);
  recNode = recCtx.createScriptProcessor(4096, 1, 1);
  recChunks = [];
  recNode.onaudioprocess = (e) => {
    recChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  recSource.connect(recNode);
  const mute = recCtx.createGain();
  mute.gain.value = 0;
  recNode.connect(mute);
  mute.connect(recCtx.destination);
  recBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = "Đang thu… nói một từ rồi bấm Dừng";
});

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  recBtn.disabled = false;
  const sr = recCtx.sampleRate || 16000;
  recNode.disconnect();
  recSource.disconnect();
  recStream.getTracks().forEach((t) => t.stop());
  await recCtx.close();
  const len = recChunks.reduce((a, c) => a + c.length, 0);
  const samples = new Float32Array(len);
  let o = 0;
  recChunks.forEach((c) => {
    samples.set(c, o);
    o += c.length;
  });
  const blob = encodeWav(samples, sr);
  sendBlob(blob, "mic.wav");
});

fetch("/api/metrics")
  .then((r) => (r.ok ? r.json() : null))
  .then((m) => {
    if (!m) return;
    const rows = (m.results || [])
      .map((x) => `<tr><td>${x.name}</td><td>${(x.accuracy * 100).toFixed(1)}%</td></tr>`)
      .join("");
    const extra = m.accuracy_new_speaker != null
      ? `<p class="meta">Giọng quen ${(m.accuracy_known_speaker * 100).toFixed(1)}% · giọng mới ${(m.accuracy_new_speaker * 100).toFixed(1)}%</p>`
      : "";
    $("metrics").innerHTML = `
      <h2>3. Độ chính xác lần huấn luyện</h2>
      <table><thead><tr><th>Thuật toán</th><th>Accuracy</th></tr></thead><tbody>${rows}</tbody></table>
      ${extra}
    `;
  })
  .catch(() => {});
