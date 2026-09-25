/**
 * 文件名: main.js
 * 功能: FaceAnnotator v2.0 — 前端交互
 *       - 拖拽上传 / 摄像头拍照
 *       - 自动标注 + 人脸识别
 *       - 新面孔命名 + 自动归类
 *       - 双栏预览 / 数据面板 / 校验
 */

// ============================================================
// 状态
// ============================================================
const state = {
  files: {},
  activeFileId: null,
  zoomLevel: 1.0,
  cameraStream: null,
  pendingNewFaces: [],  // 等待命名的 file_id 列表
};

// ============================================================
// DOM 引用
// ============================================================
const $ = (s) => document.querySelector(s);
const dom = {
  uploadZone: $('#upload-zone'),
  fileInput: $('#file-input'),
  fileList: $('#file-list'),
  fileListEmpty: $('#file-list-empty'),
  fileCount: $('#file-count'),
  fileSummary: $('#file-summary'),
  btnCamera: $('#btn-camera'),
  btnAnnotateAll: $('#btn-annotate-all'),
  btnDownloadAll: $('#btn-download-all'),
  btnClearAll: $('#btn-clear-all'),
  btnShutdown: $('#btn-shutdown'),
  btnCleanAll: $('#btn-clean-all'),
  btnCorrectAll: $('#btn-correct-all'),
  importUrl: $('#import-url'),
  btnImportUrl: $('#btn-import-url'),
  previewToolbar: $('#preview-toolbar'),
  toolbarFilename: $('#toolbar-filename'),
  toolbarPerson: $('#toolbar-person'),
  btnZoomIn: $('#btn-zoom-in'),
  btnZoomOut: $('#btn-zoom-out'),
  btnZoomFit: $('#btn-zoom-fit'),
  zoomText: $('#zoom-text'),
  btnExportToggle: $('#btn-export-toggle'),
  exportMenu: $('#export-menu'),
  btnDownloadViz: $('#btn-download-viz'),
  imgOriginal: $('#img-original'),
  imgAnnotated: $('#img-annotated'),
  placeholderOriginal: $('#placeholder-original'),
  placeholderAnnotated: $('#placeholder-annotated'),
  annotatedBadge: $('#annotated-badge'),
  dataPanel: $('#data-panel'),
  dataBadge: $('#data-badge'),
  dataGrid: $('#data-grid'),
  btnValidate: $('#btn-validate'),
  btnQuality: $('#btn-quality'),
  btnQualityReport: $('#btn-quality-report'),
  btnCorrect: $('#btn-correct'),
  validationResult: $('#validation-result'),
  qualityBadge: $('#quality-badge'),
  // 摄像头
  modalCamera: $('#modal-camera'),
  cameraVideo: $('#camera-video'),
  cameraCanvas: $('#camera-canvas'),
  cameraOverlay: $('#camera-overlay'),
  cameraError: $('#camera-error'),
  cameraCapture: $('#camera-capture'),
  cameraRecord: $('#camera-record'),
  cameraStopRecord: $('#camera-stop-record'),
  cameraClose: $('#camera-close'),
  cameraCancel: $('#camera-cancel'),
  cameraTitle: $('#camera-title'),
  cameraRecIndicator: $('#camera-rec-indicator'),
  // 命名
  modalName: $('#modal-name'),
  namePreviewImg: $('#name-preview-img'),
  nameInput: $('#name-input'),
  nameConfirm: $('#name-confirm'),
  nameSkip: $('#name-skip'),
  nameHint: $('#name-hint'),
  // 确认
  modalConfirm: $('#modal-confirm'),
  modalText: $('#modal-text'),
  modalCancel: $('#modal-cancel'),
  modalConfirmBtn: $('#modal-confirm-btn'),
};

// ============================================================
// 工具函数
// ============================================================

function showToast(msg, type = 'info') {
  const container = dom.toastContainer || document.querySelector('#toast-container');
  if (!container) { console.log('[Toast]', msg); return; }
  const el = document.createElement('div');
  el.className = `toast toast--${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function updateStats() {
  const files = Object.values(state.files);
  dom.fileCount.textContent = files.length;
  const completed = files.filter(f => f.status === 'completed').length;
  const named = files.filter(f => !!f.person_name).length;
  dom.fileSummary.textContent = `${completed} 已标注 | ${named} 已命名`;
  dom.btnAnnotateAll.disabled = files.length === 0;
  dom.btnDownloadAll.disabled = completed === 0;
}


// ============================================================
// 文件上传
// ============================================================

dom.fileInput.addEventListener('change', () => {
  if (dom.fileInput.files.length > 0) {
    uploadFiles(dom.fileInput.files);
    dom.fileInput.value = '';
  }
});

dom.uploadZone.addEventListener('click', () => dom.fileInput.click());

dom.uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dom.uploadZone.classList.add('upload-zone--drag');
});

dom.uploadZone.addEventListener('dragleave', () => {
  dom.uploadZone.classList.remove('upload-zone--drag');
});

dom.uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dom.uploadZone.classList.remove('upload-zone--drag');
  if (e.dataTransfer.files.length > 0) {
    uploadFiles(e.dataTransfer.files);
  }
});

async function uploadFiles(fileList) {
  const form = new FormData();
  for (const f of fileList) {
    form.append('files', f);
  }

  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await resp.json();
    for (const f of data.files) {
      if (f.status === 'uploaded') {
        state.files[f.file_id] = {
          file_id: f.file_id,
          filename: f.filename,
          status: 'uploaded',
          person_name: null,
          is_new_face: false,
        };
      } else {
        showToast(`${f.filename}: ${f.error}`, 'error');
      }
    }
    renderFileList();
    updateStats();
  } catch (e) {
    showToast('上传失败: ' + e.message, 'error');
  }
}

// ============================================================
// 摄像头 — 拍照 + 录像 + 实时识别 + 自动截图
// ============================================================

let camRecording = false;
let camDetectTimer = null;
let camCapturedFaces = [];    // {blob, bbox, name, is_new, confidence}
let camLastFaceKey = null;    // 用于判断角度变化
const CAM_DETECT_INTERVAL = 600;   // 600ms 检测一帧
const CAM_FRAME_W = 200;           // 极小尺寸帧，极速传输+推理

dom.btnCamera.addEventListener('click', openCamera);
dom.cameraClose.addEventListener('click', closeCamera);
dom.cameraCancel.addEventListener('click', closeCamera);
dom.cameraCapture.addEventListener('click', capturePhoto);
dom.cameraRecord.addEventListener('click', startRecording);
dom.cameraStopRecord.addEventListener('click', stopRecording);

async function openCamera() {
  dom.modalCamera.style.display = 'flex';
  dom.cameraError.style.display = 'none';
  dom.cameraCapture.disabled = true;
  dom.cameraRecord.style.display = 'none';
  dom.cameraStopRecord.style.display = 'none';
  dom.cameraOverlay.style.display = 'none';
  dom.cameraTitle.textContent = '\u{1F4F7} 拍照';
  dom.cameraRecIndicator.style.display = 'none';

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    state.cameraStream = stream;
    dom.cameraVideo.srcObject = stream;

    await new Promise(r => { dom.cameraVideo.onloadedmetadata = r; });

    // 初始化 overlay canvas
    dom.cameraOverlay.width = dom.cameraVideo.videoWidth;
    dom.cameraOverlay.height = dom.cameraVideo.videoHeight;
    dom.cameraOverlay.style.width = dom.cameraVideo.style.width || '100%';
    dom.cameraOverlay.style.height = dom.cameraVideo.style.height || 'auto';
    dom.cameraOverlay.style.display = 'block';

    dom.cameraCapture.disabled = false;
    dom.cameraRecord.style.display = 'inline-flex';
    dom.cameraRecord.disabled = false;
  } catch (e) {
    dom.cameraError.style.display = 'block';
    console.error('Camera error:', e);
  }
}

function closeCamera() {
  if (camRecording) stopRecording();
  if (camDetectTimer) { clearTimeout(camDetectTimer); camDetectTimer = null; }
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
  dom.cameraVideo.srcObject = null;
  dom.cameraOverlay.style.display = 'none';
  dom.modalCamera.style.display = 'none';
  camCapturedFaces = [];
  camLastFaceKey = null;
}

function capturePhoto() {
  const video = dom.cameraVideo;
  const canvas = dom.cameraCanvas;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  canvas.toBlob(async (blob) => {
    const form = new FormData();
    form.append('file', blob, 'camera.jpg');

    try {
      const resp = await fetch('/api/capture', { method: 'POST', body: form });
      const data = await resp.json();
      if (data.status === 'uploaded') {
        state.files[data.file_id] = {
          file_id: data.file_id,
          filename: data.filename,
          status: 'uploaded',
          person_name: null,
          is_new_face: false,
        };
        renderFileList();
        updateStats();
        showToast('拍照上传成功', 'success');
      }
    } catch (e) {
      showToast('拍照上传失败: ' + e.message, 'error');
    }
    closeCamera();
  }, 'image/jpeg', 0.92);
}

// ---- 录像模式 ----

async function startRecording() {
  camRecording = true;
  camCapturedFaces = [];
  camLastFaceKey = null;

  dom.cameraTitle.textContent = '\u{1F3A5} 录像中';
  dom.cameraRecIndicator.style.display = 'inline';
  dom.cameraCapture.style.display = 'none';
  dom.cameraRecord.style.display = 'none';
  dom.cameraStopRecord.style.display = 'inline-flex';
  dom.cameraOverlay.style.display = 'block';

  detectFrame();
}

function stopRecording() {
  camRecording = false;
  if (camDetectTimer) { clearTimeout(camDetectTimer); camDetectTimer = null; }

  dom.cameraTitle.textContent = '\u{1F4F7} 拍照';
  dom.cameraRecIndicator.style.display = 'none';
  dom.cameraCapture.style.display = 'inline-flex';
  dom.cameraRecord.style.display = 'inline-flex';
  dom.cameraStopRecord.style.display = 'none';

  // 清除 overlay 上的框
  const ctx = dom.cameraOverlay.getContext('2d');
  ctx.clearRect(0, 0, dom.cameraOverlay.width, dom.cameraOverlay.height);

  // 将截图加入文件列表
  addCapturesToFileList();
}

function detectFrame() {
  if (!camRecording) return;

  const video = dom.cameraVideo;
  const canvas = dom.cameraCanvas;

  // 缩小帧到 320px 宽，大幅减少传输 + 推理时间
  const scale = CAM_FRAME_W / video.videoWidth;
  canvas.width = CAM_FRAME_W;
  canvas.height = Math.round(video.videoHeight * scale);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (!camRecording) return;

    const form = new FormData();
    form.append('file', blob, 'frame.jpg');

    try {
      const resp = await fetch('/api/detect-frame', { method: 'POST', body: form });
      const data = await resp.json();

      // 把 bbox 坐标映射回 overlay 原始尺寸
      const scaleX = dom.cameraOverlay.width / canvas.width;
      const scaleY = dom.cameraOverlay.height / canvas.height;
      for (const face of data.faces) {
        face.bbox.x = Math.round(face.bbox.x * scaleX);
        face.bbox.y = Math.round(face.bbox.y * scaleY);
        face.bbox.width = Math.round(face.bbox.width * scaleX);
        face.bbox.height = Math.round(face.bbox.height * scaleY);
      }

      drawFacesOnOverlay(data.faces);

      // 自动截图：有新面孔或角度变化较大
      for (const face of data.faces) {
        const faceKey = face.name || 'new';
        const posKey = `${Math.round(face.bbox.x/50)},${Math.round(face.bbox.y/50)},${Math.round(face.bbox.width/50)}`;
        const fullKey = `${faceKey}_${posKey}`;

        if (fullKey !== camLastFaceKey || face.is_new) {
          // 新角度或新面孔 → 截图
          const snapCanvas = document.createElement('canvas');
          snapCanvas.width = video.videoWidth;
          snapCanvas.height = video.videoHeight;
          const sctx = snapCanvas.getContext('2d');
          sctx.drawImage(video, 0, 0);

          snapCanvas.toBlob((snapBlob) => {
            camCapturedFaces.push({
              blob: snapBlob,
              bbox: face.bbox,
              name: face.name,
              is_new: face.is_new,
              confidence: face.confidence,
            });
          }, 'image/jpeg', 0.88);

          camLastFaceKey = fullKey;
          break;  // 一帧最多截一张
        }
      }
    } catch (e) {
      // 静默失败，继续下一帧
    }

    if (camRecording) {
      camDetectTimer = setTimeout(detectFrame, CAM_DETECT_INTERVAL);
    }
  }, 'image/jpeg', 0.45);  // 低质量 JPEG，减小传输
}

function drawFacesOnOverlay(faces) {
  const canvas = dom.cameraOverlay;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const face of faces) {
    const { x, y, width, height } = face.bbox;

    // 边框颜色：已命名=绿色，新面孔=黄色
    const color = face.name ? '#00E5FF' : '#FFAB00';
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, width, height);

    // 标签背景
    const label = face.name || '新';  // "新"
    const confStr = face.confidence > 0 ? ` ${(face.confidence*100).toFixed(0)}%` : '';
    const text = face.name ? `${label}${confStr}` : label;
    ctx.font = 'bold 16px Inter, sans-serif';
    const tm = ctx.measureText(text);
    const th = 20;
    const pad = 6;

    ctx.fillStyle = color;
    ctx.fillRect(x, y - th - 4, tm.width + pad * 2, th + 4);
    ctx.fillStyle = '#000';
    ctx.fillText(text, x + pad, y - 8);
  }
}

function addCapturesToFileList() {
  if (camCapturedFaces.length === 0) {
    showToast('未捕获到人脸截图', 'info');
    return;
  }

  let addedNew = 0;
  let addedNamed = 0;

  // 异步处理所有截图
  const uploadPromises = camCapturedFaces.map(async (cap, idx) => {
    const form = new FormData();
    form.append('file', cap.blob, `video_${idx}.jpg`);
    const resp = await fetch('/api/capture', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.status === 'uploaded') {
      state.files[data.file_id] = {
        file_id: data.file_id,
        filename: `video_cap_${idx + 1}.jpg`,
        status: 'uploaded',
        person_name: cap.name || null,
        is_new_face: cap.is_new,
      };
      if (cap.is_new) addedNew++;
      else if (cap.name) addedNamed++;
    }
    return data;
  });

  Promise.all(uploadPromises).then(() => {
    renderFileList();
    updateStats();
    const parts = [];
    if (addedNew > 0) parts.push(`${addedNew} 张新面孔`);
    if (addedNamed > 0) parts.push(`${addedNamed} 张已识别`);
    showToast(`视频截图: ${parts.join(', ')} 共 ${camCapturedFaces.length} 张，请点击"全部标注"`, 'success');
  });

  camCapturedFaces = [];
}

// ============================================================
// 文件列表渲染
// ============================================================

function renderFileList() {
  const files = Object.values(state.files);
  dom.fileList.innerHTML = '';
  dom.fileListEmpty.style.display = files.length === 0 ? 'flex' : 'none';

  for (const f of files) {
    const item = document.createElement('div');
    item.className = `file-item ${f.file_id === state.activeFileId ? 'file-item--active' : ''}`;
    item.dataset.fileId = f.file_id;

    // 状态指示器：文字 + 颜色，清楚直观
    let statusHtml = '';
    if (f.status === 'processing') {
      statusHtml = '<span class="file-item__spinner"></span>';
    } else if (f.status === 'completed') {
      statusHtml = '<span class="file-item__status file-item__status--done" title="标注完成">&#10003;</span>';
    } else if (f.status === 'failed') {
      statusHtml = '<span class="file-item__status file-item__status--error" title="标注失败">&#10007;</span>';
    } else {
      statusHtml = '<span class="file-item__status" title="待标注">&#9679;</span>';
    }

    // 状态文字标签
    let statusLabel = '';
    if (f.status === 'processing') {
      statusLabel = '<span class="file-item__processing-text">标注中...</span>';
    } else if (f.status === 'failed') {
      statusLabel = '<span class="file-item__status-label file-item__status-label--error">失败</span>';
    } else if (f.status === 'uploaded') {
      statusLabel = '<span class="file-item__status-label">待标注</span>';
    }

    // 每张图单独的「标注」按钮（未完成的才显示）
    const annotateBtn = (f.status === 'uploaded' || f.status === 'failed')
      ? `<button class="file-item__annotate-btn" data-action="annotate" data-file-id="${f.file_id}" title="标注此图">标注</button>`
      : '';

    item.innerHTML = `
      ${statusHtml}
      <span class="file-item__name" title="${f.filename}">${f.filename}</span>
      ${statusLabel}
      ${f.person_name ? `<span class="file-item__person" title="已命名: ${f.person_name}">&#128100; ${f.person_name}</span>` : ''}
      ${f.is_new_face ? `<span class="file-item__new-tag" data-action="name" data-file-id="${f.file_id}" title="点击命名">新面孔</span>` : ''}
      ${annotateBtn}
      <button class="file-item__del" data-action="delete" title="删除">&times;</button>
    `;

    item.addEventListener('click', async (e) => {
      if (e.target.dataset.action === 'delete') {
        e.stopPropagation();
        deleteFile(f.file_id);
        return;
      }
      if (e.target.dataset.action === 'annotate') {
        e.stopPropagation();
        const fid = e.target.dataset.fileId;
        await annotateFile(fid);
        processPendingNames();
        return;
      }
      if (e.target.dataset.action === 'name') {
        e.stopPropagation();
        const fid = e.target.dataset.fileId;
        promptName(fid);
        return;
      }
      selectFile(f.file_id);
    });

    dom.fileList.appendChild(item);
  }
}

async function selectFile(fileId) {
  state.activeFileId = fileId;
  const f = state.files[fileId];
  if (!f) return;

  dom.toolbarFilename.textContent = f.filename;
  dom.toolbarPerson.textContent = f.person_name ? `— ${f.person_name}` : '';
  dom.toolbarPerson.style.display = f.person_name ? 'inline' : 'none';

  // 显示原图
  dom.imgOriginal.src = `/api/image/${fileId}?t=${Date.now()}`;
  dom.imgOriginal.style.display = 'block';
  dom.placeholderOriginal.style.display = 'none';

  // 标注图
  if (f.status === 'completed') {
    dom.imgAnnotated.src = `/api/viz/${fileId}?t=${Date.now()}`;
    dom.imgAnnotated.style.display = 'block';
    dom.placeholderAnnotated.style.display = 'none';
    dom.annotatedBadge.style.display = 'inline';
    dom.annotatedBadge.textContent = f.person_name || '已标注';
    dom.annotatedBadge.className = f.person_name
      ? 'preview-pane__badge preview-pane__badge--named'
      : 'preview-pane__badge';
    showAnnotationData(fileId);
  } else {
    dom.imgAnnotated.style.display = 'none';
    dom.placeholderAnnotated.style.display = 'flex';
    dom.annotatedBadge.style.display = 'none';
    dom.dataPanel.style.display = 'none';
  }

  renderFileList();
  updateStats();
}

function showAnnotationData(fileId) {
  const f = state.files[fileId];
  if (!f || !f.annotation) {
    dom.dataPanel.style.display = 'none';
    return;
  }

  dom.dataPanel.style.display = 'block';
  const ann = f.annotation;
  dom.dataBadge.textContent = `标注者: ${ann.annotator}`;
  if (dom.btnValidate) dom.btnValidate.style.display = 'inline-block';
  if (dom.btnQuality) dom.btnQuality.style.display = 'inline-block';
  if (dom.btnQualityReport) dom.btnQualityReport.style.display = 'inline-block';
  if (dom.btnCorrect) dom.btnCorrect.style.display = 'inline-block';

  // 显示质量评分
  showQualityBadge(ann);

  dom.dataGrid.innerHTML = `
    <div class="data-item">
      <span class="data-item__label">边界框 (bbox)</span>
      <span class="data-item__value">x=${ann.bbox?.x} y=${ann.bbox?.y} w=${ann.bbox?.width} h=${ann.bbox?.height}</span>
    </div>
    <div class="data-item">
      <span class="data-item__label">左眼</span>
      <span class="data-item__value">[${ann.landmarks?.left_eye}]</span>
    </div>
    <div class="data-item">
      <span class="data-item__label">右眼</span>
      <span class="data-item__value">[${ann.landmarks?.right_eye}]</span>
    </div>
    <div class="data-item">
      <span class="data-item__label">鼻尖</span>
      <span class="data-item__value">[${ann.landmarks?.nose_tip}]</span>
    </div>
    <div class="data-item">
      <span class="data-item__label">左嘴角</span>
      <span class="data-item__value">[${ann.landmarks?.left_mouth_corner}]</span>
    </div>
    <div class="data-item">
      <span class="data-item__label">右嘴角</span>
      <span class="data-item__value">[${ann.landmarks?.right_mouth_corner}]</span>
    </div>
    ${ann.confidence ? `<div class="data-item">
      <span class="data-item__label">置信度</span>
      <span class="data-item__value">${(ann.confidence * 100).toFixed(1)}%</span>
    </div>` : ''}
    ${ann.quality ? `<div class="data-item data-item--quality">
      <span class="data-item__label">质量评分</span>
      <span class="data-item__value">${(ann.quality.quality_score * 100).toFixed(0)}% (${ann.quality.quality_label === 'good' ? '优秀' : ann.quality.quality_label === 'acceptable' ? '可接受' : '较差'})</span>
    </div>
    <div class="data-item data-item--sub">
      <span class="data-item__label">模糊度</span>
      <span class="data-item__value">${(ann.quality.blur_score * 100).toFixed(0)}%</span>
    </div>
    <div class="data-item data-item--sub">
      <span class="data-item__label">亮度</span>
      <span class="data-item__value">${(ann.quality.brightness_score * 100).toFixed(0)}%</span>
    </div>
    <div class="data-item data-item--sub">
      <span class="data-item__label">姿态</span>
      <span class="data-item__value">${(ann.quality.pose_score * 100).toFixed(0)}%</span>
    </div>` : ''}
  `;
}

function showQualityBadge(ann) {
  if (!dom.qualityBadge) return;
  if (!ann.quality) {
    dom.qualityBadge.style.display = 'none';
    return;
  }
  const q = ann.quality;
  dom.qualityBadge.style.display = 'block';
  const labelText = q.quality_label === 'good' ? '优秀' : q.quality_label === 'acceptable' ? '可接受' : '较差';
  const labelClass = `quality-badge--${q.quality_label}`;
  dom.qualityBadge.innerHTML = `
    <div class="quality-score ${labelClass}">
      <span class="quality-score__label">图片质量</span>
      <span class="quality-score__value">${(q.quality_score * 100).toFixed(0)}%</span>
      <span class="quality-score__tag">${labelText}</span>
    </div>
    <div class="quality-details">
      <div class="quality-detail"><span>模糊</span><span>${(q.blur_score * 100).toFixed(0)}%</span></div>
      <div class="quality-detail"><span>亮度</span><span>${(q.brightness_score * 100).toFixed(0)}%</span></div>
      <div class="quality-detail"><span>姿态</span><span>${(q.pose_score * 100).toFixed(0)}%</span></div>
    </div>
  `;
}

// ============================================================
// 标注 + 识别
// ============================================================

let annotatingLock = false;

dom.btnAnnotateAll.addEventListener('click', annotateAll);

async function annotateAll() {
  if (annotatingLock) return;  // 防止重复触发
  annotatingLock = true;

  const files = Object.values(state.files).filter(f => f.status === 'uploaded');

  for (const f of files) {
    await annotateFile(f.file_id);
  }

  annotatingLock = false;
  // 全部标注完成后，才启动命名队列
  processPendingNames();
}

async function annotateFile(fileId) {
  if (!state.files[fileId]) return;
  // 已完成的不再重复标注
  if (state.files[fileId].status === 'completed') {
    console.log('[标注] 跳过已完成:', fileId);
    return;
  }

  console.log('[标注] 开始:', fileId, '当前状态:', state.files[fileId].status);
  state.files[fileId].status = 'processing';
  renderFileList();

  try {
    const resp = await fetch(`/api/annotate/${fileId}`, { method: 'POST' });
    const data = await resp.json();

    console.log('[标注] 完成:', fileId, '后端状态:', data.status);
    state.files[fileId].status = data.status;
    state.files[fileId].annotation = data.annotation;
    state.files[fileId].person_name = data.person_name;
    state.files[fileId].is_new_face = data.is_new_face;

    if (data.is_new_face) {
      state.pendingNewFaces.push(fileId);
    }

    if (fileId === state.activeFileId) {
      selectFile(fileId);
    }
    renderFileList();
    updateStats();

    try { if (data.person_name) showToast(`已识别: ${data.person_name}`, 'success'); } catch (_) {}
  } catch (e) {
    console.error('[标注] 失败:', fileId, e);
    state.files[fileId].status = 'failed';
    try { showToast(`标注失败: ${e.message}`, 'error'); } catch (_) {}
    renderFileList();
    updateStats();
  }
}

function processPendingNames() {
  // 如果命名弹窗已打开，不覆盖
  if (dom.modalName.style.display === 'flex') return;
  // 清理已失效的 fileId（文件可能已被删除）
  state.pendingNewFaces = state.pendingNewFaces.filter(fid => state.files[fid]);
  // 如果队列为空，结束
  if (state.pendingNewFaces.length === 0) return;

  const fileId = state.pendingNewFaces.shift();
  console.log('[命名队列] 弹出:', fileId, '剩余:', state.pendingNewFaces.length);
  promptName(fileId);
}

function promptName(fileId) {
  const f = state.files[fileId];
  if (!f) {
    // 该文件可能已被删除，跳过，继续处理下一个
    processPendingNames();
    return;
  }

  dom.namePreviewImg.src = `/api/image/${fileId}`;
  dom.nameInput.value = '';
  dom.nameConfirm.disabled = true;
  dom.nameHint.textContent = '';
  dom.modalName.style.display = 'flex';
  dom.nameInput.focus();

  // 存储当前正在命名的 fileId
  dom.nameInput.dataset.fileId = fileId;
}

async function confirmName() {
  const fileId = dom.nameInput.dataset.fileId;
  const name = dom.nameInput.value.trim();
  if (!name || !fileId) return;

  dom.nameConfirm.disabled = true;
  dom.nameConfirm.textContent = '...';

  const form = new FormData();
  form.append('file_id', fileId);
  form.append('name', name);

  try {
    const resp = await fetch('/api/person/name', { method: 'POST', body: form });
    const data = await resp.json();
    state.files[fileId].person_name = data.person_name;
    state.files[fileId].is_new_face = false;
    renderFileList();
    updateStats();
    if (fileId === state.activeFileId) selectFile(fileId);
    showToast(`已命名: ${data.person_name}`, 'success');
  } catch (e) {
    showToast('命名失败: ' + e.message, 'error');
  }

  dom.nameConfirm.textContent = '确认';
  dom.nameConfirm.disabled = false;
  dom.modalName.style.display = 'none';
  // 处理队列中的下一个
  setTimeout(() => processPendingNames(), 200);
}

function skipName() {
  const fileId = dom.nameInput.dataset.fileId;
  if (fileId && state.files[fileId]) {
    state.files[fileId].is_new_face = false;
    renderFileList();
    updateStats();
  }
  dom.modalName.style.display = 'none';
  // 处理队列中的下一个
  setTimeout(() => processPendingNames(), 200);
}

dom.nameConfirm.addEventListener('click', confirmName);
dom.nameSkip.addEventListener('click', skipName);
dom.nameInput.addEventListener('input', () => {
  dom.nameConfirm.disabled = dom.nameInput.value.trim().length === 0;
});
dom.nameInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && dom.nameInput.value.trim()) {
    confirmName();
  }
});

// ============================================================
// 校验
// ============================================================

dom.btnValidate.addEventListener('click', async () => {
  const fileId = state.activeFileId;
  if (!fileId) return;

  try {
    const resp = await fetch(`/api/validate/${fileId}`, { method: 'POST' });
    const data = await resp.json();
    dom.validationResult.style.display = 'block';

    if (data.errors.length === 0 && data.warnings.length === 0) {
      dom.validationResult.innerHTML = '<div class="validation-ok">全部校验通过</div>';
    } else {
      let html = '';
      for (const e of data.errors) {
        html += `<div class="validation-error">${e}</div>`;
      }
      for (const w of data.warnings) {
        html += `<div class="validation-warning">${w}</div>`;
      }
      dom.validationResult.innerHTML = html;
    }
  } catch (e) {
    showToast('校验失败: ' + e.message, 'error');
  }
});

// ============================================================
// 导出下拉菜单
// ============================================================

if (dom.btnExportToggle) {
  dom.btnExportToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    if (dom.exportMenu) {
      dom.exportMenu.style.display = dom.exportMenu.style.display === 'none' ? 'block' : 'none';
    }
  });
}

// 点击其他区域关闭菜单
document.addEventListener('click', () => {
  if (dom.exportMenu) dom.exportMenu.style.display = 'none';
});

// 导出菜单项点击
if (dom.exportMenu) {
  dom.exportMenu.querySelectorAll('.toolbar-dropdown__item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.stopPropagation();
      if (dom.exportMenu) dom.exportMenu.style.display = 'none';
      const format = item.dataset.format;

      if (format === 'json') {
        if (state.activeFileId) window.open(`/api/download/${state.activeFileId}`);
      } else if (format === 'json-all') {
        window.open('/api/download-all');
      } else if (format === 'coco') {
        if (state.activeFileId) window.open(`/api/export/${state.activeFileId}?format=coco`);
      } else if (format === 'yolo') {
        if (state.activeFileId) window.open(`/api/export/${state.activeFileId}?format=yolo`);
      } else if (format === 'voc') {
        if (state.activeFileId) window.open(`/api/export/${state.activeFileId}?format=voc`);
      } else if (format === 'csv') {
        if (state.activeFileId) window.open(`/api/export/${state.activeFileId}?format=csv`);
      } else if (format === 'coco-all') {
        window.open('/api/export-all?format=coco');
      } else if (format === 'yolo-all') {
        window.open('/api/export-all?format=yolo');
      } else if (format === 'voc-all') {
        window.open('/api/export-all?format=voc');
      } else if (format === 'csv-all') {
        window.open('/api/export-all?format=csv');
      }
    });
  });
}

// ============================================================
// 下载
// ============================================================

dom.btnDownloadViz.addEventListener('click', () => {
  if (state.activeFileId) window.open(`/api/download-viz/${state.activeFileId}`);
});

dom.btnDownloadAll.addEventListener('click', () => {
  window.open('/api/download-all');
});

// ============================================================
// 质量评估
// ============================================================

if (dom.btnQuality) {
  dom.btnQuality.addEventListener('click', async () => {
    const fileId = state.activeFileId;
  if (!fileId) return;

  dom.btnQuality.textContent = '...';
  dom.btnQuality.disabled = true;

  try {
    const resp = await fetch(`/api/quality/${fileId}`, { method: 'POST' });
    const data = await resp.json();

    if (data.status === 'valid' && data.quality) {
      // 更新本地状态
      if (state.files[fileId] && state.files[fileId].annotation) {
        state.files[fileId].annotation.quality = {
          quality_score: data.quality.quality_score,
          quality_label: data.quality.quality_label,
          blur_score: data.quality.image_quality.blur_score,
          brightness_score: data.quality.image_quality.brightness_score,
          pose_score: data.quality.image_quality.pose_score,
        };
        showAnnotationData(fileId);
      }
      const label = data.quality.quality_label === 'good' ? '优秀' : data.quality.quality_label === 'acceptable' ? '可接受' : '较差';
      showToast(`质量评估: ${(data.quality.quality_score * 100).toFixed(0)}% - ${label}`, 'success');
    } else if (data.status === 'invalid') {
      showToast('标注校验未通过，无法评估质量', 'error');
    }
  } catch (e) {
    showToast('质量评估失败: ' + e.message, 'error');
  }

  if (dom.btnQuality) { dom.btnQuality.textContent = '★ 质量评估'; dom.btnQuality.disabled = false; }
  });
}

// 质量报告按钮
if (dom.btnQualityReport) {
  dom.btnQualityReport.addEventListener('click', async () => {
  dom.btnQualityReport.textContent = '...';
  dom.btnQualityReport.disabled = true;

  try {
    const resp = await fetch('/api/quality-report');
    const data = await resp.json();

    const s = data.summary;
    const qd = s.quality_distribution;
    const sub = s.sub_scores;

    let reportHtml = '<div class="quality-report-popup">';
    reportHtml += '<h4>质量评估报告</h4>';
    reportHtml += `<p>评估文件: ${data.total} 个 | 平均分: ${(qd.mean_score * 100).toFixed(0)}%</p>`;
    reportHtml += `<div class="quality-report-grid">`;
    reportHtml += `<div><span>模糊度均分</span><b>${(sub.mean_blur * 100).toFixed(0)}%</b></div>`;
    reportHtml += `<div><span>亮度均分</span><b>${(sub.mean_brightness * 100).toFixed(0)}%</b></div>`;
    reportHtml += `<div><span>姿态均分</span><b>${(sub.mean_pose * 100).toFixed(0)}%</b></div>`;
    reportHtml += `<div><span>人脸占比均分</span><b>${(sub.mean_face_ratio * 100).toFixed(0)}%</b></div>`;
    reportHtml += `<div><span>关键点位置均分</span><b>${(sub.mean_landmark_position * 100).toFixed(0)}%</b></div>`;
    reportHtml += `</div>`;
    reportHtml += `<p>等级分布: `;
    for (const [label, count] of Object.entries(qd.by_label || {})) {
      const l = label === 'good' ? '优秀' : label === 'acceptable' ? '可接受' : '较差';
      reportHtml += `${l} ${count}件 | `;
    }
    reportHtml += `</p>`;
    if (s.poor_files_count > 0) {
      reportHtml += `<p class="quality-report-warn">建议复核: ${s.review_files_count} 件 | 建议拒绝: ${s.poor_files_count} 件</p>`;
    }
    reportHtml += '</div>';

    dom.validationResult.style.display = 'block';
    dom.validationResult.innerHTML = reportHtml;
  } catch (e) {
    showToast('质量报告生成失败: ' + e.message, 'error');
  }

  if (dom.btnQualityReport) { dom.btnQualityReport.textContent = '☰ 质量报告'; dom.btnQualityReport.disabled = false; }
  });
}

// ============================================================
// 图片清洗
// ============================================================

if (dom.btnCleanAll) {
  dom.btnCleanAll.addEventListener('click', async () => {
    const files = Object.values(state.files).filter(f => f.status === 'uploaded');
    if (!files.length) { showToast('没有待清洗的文件', 'info'); return; }

    dom.btnCleanAll.disabled = true;
    let count = 0;
    for (const f of files) {
      dom.btnCleanAll.textContent = '清洗中 ' + (++count) + '/' + files.length + '...';
      try {
        const resp = await fetch('/api/preprocess/' + f.file_id, { method: 'POST' });
        if (resp.ok) showToast('已清洗: ' + f.filename, 'success');
      } catch (e) {
        showToast('清洗失败: ' + f.filename, 'error');
      }
    }
    dom.btnCleanAll.textContent = '★ 全部清洗';
    dom.btnCleanAll.disabled = false;
  });
}

// ============================================================
// 标注纠错
// ============================================================

if (dom.btnCorrect) {
  dom.btnCorrect.addEventListener('click', async () => {
    const fileId = state.activeFileId;
    if (!fileId) return;

    dom.btnCorrect.textContent = '...';
    dom.btnCorrect.disabled = true;
    try {
      const resp = await fetch('/api/correct/' + fileId, { method: 'POST' });
      const data = await resp.json();
      if (data.corrections > 0) {
        state.files[fileId].annotation = data.modifications;
        selectFile(fileId);
      }
      showToast('纠错: ' + data.corrections + ' 处修正', 'success');
    } catch (e) {
      showToast('纠错失败: ' + e.message, 'error');
    }
    dom.btnCorrect.textContent = '☡ 纠错';
    dom.btnCorrect.disabled = false;
  });
}

if (dom.btnCorrectAll) {
  dom.btnCorrectAll.addEventListener('click', async () => {
    const files = Object.values(state.files).filter(f => f.status === 'completed');
    if (!files.length) { showToast('没有已标注的文件', 'info'); return; }

    dom.btnCorrectAll.disabled = true;
    let count = 0, totalCorrections = 0;
    for (const f of files) {
      dom.btnCorrectAll.textContent = '纠错中 ' + (++count) + '/' + files.length + '...';
      try {
        const resp = await fetch('/api/correct/' + f.file_id, { method: 'POST' });
        const data = await resp.json();
        totalCorrections += data.corrections;
      } catch (e) { /* skip */ }
    }
    showToast('纠错完成: ' + totalCorrections + ' 处修正', 'success');
    dom.btnCorrectAll.textContent = '☡ 全部纠错';
    dom.btnCorrectAll.disabled = false;
  });
}

// ============================================================
// URL 导入
// ============================================================

if (dom.btnImportUrl && dom.importUrl) {
  dom.btnImportUrl.addEventListener('click', async () => {
    const url = dom.importUrl.value.trim();
    if (!url) { showToast('请输入URL', 'error'); return; }

    dom.btnImportUrl.disabled = true;
    dom.btnImportUrl.textContent = '...';
    try {
      const form = new FormData();
      form.append('url', url);
      const resp = await fetch('/api/import/url', { method: 'POST', body: form });
      const data = await resp.json();
      if (data.status === 'uploaded') {
        state.files[data.file_id] = {
          file_id: data.file_id,
          filename: data.filename,
          status: 'uploaded',
          person_name: null,
          is_new_face: false,
        };
        renderFileList();
        updateStats();
        dom.importUrl.value = '';
        showToast('URL导入成功: ' + data.filename, 'success');
      }
    } catch (e) {
      showToast('URL导入失败: ' + e.message, 'error');
    }
    dom.btnImportUrl.textContent = '导入';
    dom.btnImportUrl.disabled = false;
  });

  dom.importUrl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') dom.btnImportUrl.click();
  });
}

// 扩展 updateStats
(function() {
  const orig = updateStats;
  updateStats = function() {
    orig();
    const uploaded = Object.values(state.files).filter(f => f.status === 'uploaded').length;
    const completed = Object.values(state.files).filter(f => f.status === 'completed').length;
    if (dom.btnCleanAll) dom.btnCleanAll.disabled = uploaded === 0;
    if (dom.btnCorrectAll) dom.btnCorrectAll.disabled = completed === 0;
  };
})();

// ============================================================
// 删除 / 清空
// ============================================================

async function deleteFile(fileId) {
  try {
    await fetch(`/api/file/${fileId}`, { method: 'DELETE' });
    delete state.files[fileId];
    if (state.activeFileId === fileId) {
      state.activeFileId = null;
      dom.imgOriginal.style.display = 'none';
      dom.imgAnnotated.style.display = 'none';
      dom.placeholderOriginal.style.display = 'flex';
      dom.placeholderAnnotated.style.display = 'flex';
      dom.dataPanel.style.display = 'none';
      dom.toolbarFilename.textContent = '预览';
      dom.toolbarPerson.style.display = 'none';
    }
    renderFileList();
    updateStats();
  } catch (e) {
    showToast('删除失败', 'error');
  }
}

dom.btnClearAll.addEventListener('click', () => {
  dom.modalText.textContent = '确认清空所有上传文件和标注数据？';
  dom.modalConfirm.style.display = 'flex';
  dom.modalConfirmBtn.onclick = async () => {
    try {
      await fetch('/api/clear', { method: 'POST' });
      state.files = {};
      state.activeFileId = null;
      dom.imgOriginal.style.display = 'none';
      dom.imgAnnotated.style.display = 'none';
      dom.placeholderOriginal.style.display = 'flex';
      dom.placeholderAnnotated.style.display = 'flex';
      dom.dataPanel.style.display = 'none';
      dom.toolbarFilename.textContent = '预览';
      dom.toolbarPerson.style.display = 'none';
      dom.imgOriginal.src = '';
      dom.imgAnnotated.src = '';
      renderFileList();
      updateStats();
      dom.modalConfirm.style.display = 'none';
      showToast('已清空', 'info');
    } catch (e) {
      showToast('清空失败', 'error');
    }
  };
});

dom.modalCancel.addEventListener('click', () => {
  dom.modalConfirm.style.display = 'none';
});

// 退出系统
dom.btnShutdown.addEventListener('click', () => {
  dom.modalText.textContent = '确认退出系统？服务器将停止运行。';
  dom.modalConfirm.style.display = 'flex';
  dom.modalConfirmBtn.onclick = async () => {
    try {
      await fetch('/api/shutdown', { method: 'POST' });
    } catch (e) {
      // 服务器已关闭，请求会失败，这是正常的
    }
    showToast('系统已退出', 'info');
    dom.modalConfirm.style.display = 'none';
    setTimeout(() => { document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#888;font-family:Inter,sans-serif;font-size:18px;">系统已退出，请关闭此页面。</div>'; }, 500);
  };
});

// ============================================================
// 缩放
// ============================================================

dom.btnZoomIn.addEventListener('click', () => {
  state.zoomLevel = Math.min(3.0, state.zoomLevel + 0.25);
  applyZoom();
});
dom.btnZoomOut.addEventListener('click', () => {
  state.zoomLevel = Math.max(0.25, state.zoomLevel - 0.25);
  applyZoom();
});
dom.btnZoomFit.addEventListener('click', () => {
  state.zoomLevel = 1.0;
  applyZoom();
});

function applyZoom() {
  dom.zoomText.textContent = Math.round(state.zoomLevel * 100) + '%';
  dom.imgOriginal.style.transform = `scale(${state.zoomLevel})`;
  dom.imgAnnotated.style.transform = `scale(${state.zoomLevel})`;
}

// ============================================================
// 键盘快捷键
// ============================================================

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (dom.modalName.style.display === 'flex') {
      skipName();
    } else if (dom.modalCamera.style.display === 'flex') {
      closeCamera();
    } else if (dom.modalConfirm.style.display === 'flex') {
      dom.modalConfirm.style.display = 'none';
    }
  }
});

// ============================================================
// 初始化
// ============================================================

updateStats();
