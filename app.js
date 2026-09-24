const LABELS = [
  "abrasion", "broken_fibers", "bubble", "buckle", "buckle_line",
  "corrosion", "crack_composite", "crack_metal", "crease", "defect",
  "delamination", "dent_composite", "dent_metal", "edge_delamination",
  "erosion", "hole", "scratch", "void"
];

const MODEL_SIZE = 640;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

let session = null;
let isProcessing = false;

const loadModelBtn = document.getElementById('loadModelBtn');
const startCameraBtn = document.getElementById('startCameraBtn');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const statusText = document.getElementById('status');
const detectionsList = document.getElementById('detectionsList');

const video = document.getElementById('webcamVideo');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');

const COLORS = LABELS.map((_, i) => `hsl(${(i * 360) / LABELS.length}, 100%, 50%)`);

// --- ШАГ 1: Загрузка модели с отслеживанием прогресса ---
loadModelBtn.addEventListener('click', async () => {
  loadModelBtn.disabled = true;
  statusText.textContent = 'Скачивание модели...';
  progressContainer.style.display = 'block';

  try {
    const response = await fetch('./model.onnx');
    if (!response.ok) throw new Error(`Ошибка загрузки: ${response.statusText}`);

    const contentLength = response.headers.get('content-length');
    const total = parseInt(contentLength, 10) || 0;
    let loaded = 0;

    const reader = response.body.getReader();
    const chunks = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      loaded += value.length;

      if (total) {
        const percent = Math.round((loaded / total) * 100);
        progressBar.value = percent;
        progressText.textContent = `${percent}%`;
      }
    }

    const modelBuffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      modelBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    statusText.textContent = 'Инициализация ONNX Runtime...';

    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(modelBuffer.buffer, {
      executionProviders: ['webgpu', 'wasm']
    });

    statusText.textContent = 'Модель готова! Нажмите "Открыть камеру".';
    progressContainer.style.display = 'none';
    startCameraBtn.disabled = false;
  } catch (error) {
    console.error(error);
    statusText.textContent = 'Ошибка при загрузке модели. Проверьте консоль.';
    loadModelBtn.disabled = false;
  }
});

// --- ШАГ 2: Открытие камеры ---
startCameraBtn.addEventListener('click', async () => {
  startCameraBtn.disabled = true;
  statusText.textContent = 'Запрос доступа к камере...';

  try {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
    } catch (e) {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
    }

    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    statusText.textContent = 'Камера подключена. Детекция запущена.';
    requestAnimationFrame(processFrame);
  } catch (error) {
    console.error(error);
    statusText.textContent = 'Ошибка доступа к камере.';
    startCameraBtn.disabled = false;
  }
});

// --- Препроцессинг кадров с переводом в Grayscale ---
function preprocess(source) {
  const offscreen = document.createElement('canvas');
  offscreen.width = MODEL_SIZE;
  offscreen.height = MODEL_SIZE;
  const offCtx = offscreen.getContext('2d');

  offCtx.drawImage(source, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const imgData = offCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;

  const float32Data = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
  const planeSize = MODEL_SIZE * MODEL_SIZE;

  for (let i = 0; i < planeSize; i++) {
    const r = imgData[i * 4];
    const g = imgData[i * 4 + 1];
    const b = imgData[i * 4 + 2];

    const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;

    float32Data[i] = gray;
    float32Data[planeSize + i] = gray;
    float32Data[2 * planeSize + i] = gray;
  }

  return new ort.Tensor('float32', float32Data, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

// --- Постпроцессинг YOLOv8 ---
function parseOutput(data, origW, origH) {
  const anchors = 8400;
  const numClasses = 18;
  const boxes = [];

  const scaleX = origW / MODEL_SIZE;
  const scaleY = origH / MODEL_SIZE;

  for (let i = 0; i < anchors; i++) {
    let maxProb = 0;
    let classId = -1;

    for (let c = 0; c < numClasses; c++) {
      const prob = data[(4 + c) * anchors + i];
      if (prob > maxProb) {
        maxProb = prob;
        classId = c;
      }
    }

    if (maxProb >= CONF_THRESHOLD) {
      const cx = data[0 * anchors + i];
      const cy = data[1 * anchors + i];
      const w  = data[2 * anchors + i];
      const h  = data[3 * anchors + i];

      boxes.push({
        x: (cx - w / 2) * scaleX,
        y: (cy - h / 2) * scaleY,
        w: w * scaleX,
        h: h * scaleY,
        score: maxProb,
        classId,
        label: LABELS[classId]
      });
    }
  }

  return applyNMS(boxes);
}

function applyNMS(boxes) {
  boxes.sort((a, b) => b.score - a.score);
  const selected = [];

  while (boxes.length > 0) {
    const best = boxes.shift();
    selected.push(best);

    boxes = boxes.filter(box => {
      if (box.classId !== best.classId) return true;
      return calculateIoU(best, box) < IOU_THRESHOLD;
    });
  }

  return selected;
}

function calculateIoU(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);

  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const unionArea = (a.w * a.h) + (b.w * b.h) - interArea;

  return unionArea === 0 ? 0 : interArea / unionArea;
}

// --- Обновление списка детекций в панели UI ---
function updateDetectionsList(boxes) {
  detectionsList.innerHTML = '';

  if (boxes.length === 0) {
    const emptyItem = document.createElement('li');
    emptyItem.style.color = '#888';
    emptyItem.textContent = 'Дефекты не обнаружены';
    detectionsList.appendChild(emptyItem);
    return;
  }

  boxes.forEach(box => {
    const li = document.createElement('li');
    const color = COLORS[box.classId];
    
    li.innerHTML = `
      <span style="border-left: 4px solid ${color}; padding-left: 8px;">${box.label}</span>
      <strong>${(box.score * 100).toFixed(1)}%</strong>
    `;
    detectionsList.appendChild(li);
  });
}

// --- Отрисовка результатов ---
function renderResults(boxes) {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
  ctx.font = `${Math.max(14, Math.round(canvas.width / 40))}px sans-serif`;

  boxes.forEach(box => {
    const color = COLORS[box.classId];

    ctx.strokeStyle = color;
    ctx.strokeRect(box.x, box.y, box.w, box.h);

    const text = `${box.label} ${(box.score * 100).toFixed(0)}%`;
    const textWidth = ctx.measureText(text).width;
    const textHeight = parseInt(ctx.font, 10);

    ctx.fillStyle = color;
    ctx.fillRect(box.x, box.y - textHeight - 4, textWidth + 8, textHeight + 4);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, box.x + 4, box.y - 4);
  });

  updateDetectionsList(boxes);
}

// --- Цикл обработки кадров ---
async function processFrame() {
  if (isProcessing) {
    requestAnimationFrame(processFrame);
    return;
  }

  isProcessing = true;

  try {
    const tensor = preprocess(video);
    const output = await session.run({ images: tensor });
    const boxes = parseOutput(output.output0.data, canvas.width, canvas.height);

    renderResults(boxes);
  } catch (error) {
    console.error(error);
  }

  isProcessing = false;
  requestAnimationFrame(processFrame);
}
