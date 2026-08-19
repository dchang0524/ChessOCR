import { PIECE_GLYPHS, bestClass, buildBoardFen, squareName } from "./chess.js";
import {
  assignGroupLabels,
  completeLinkageClusters,
  cosineSimilarityMatrix,
} from "./grouping.js";

const ORT_VERSION = "1.22.0";
const BOARD_SIZE = 512;
const SQUARE_SIZE = 64;
const NUM_SQUARES = 64;

const elements = {
  upload: document.querySelector("#upload"),
  uploadZone: document.querySelector("#upload-zone"),
  editorSection: document.querySelector("#editor-section"),
  editor: document.querySelector("#crop-editor"),
  resetCrop: document.querySelector("#reset-crop"),
  cropSize: document.querySelector("#crop-size"),
  orientation: document.querySelector("#orientation"),
  sideToMove: document.querySelector("#side-to-move"),
  threshold: document.querySelector("#threshold"),
  thresholdValue: document.querySelector("#threshold-value"),
  recognize: document.querySelector("#recognize"),
  modelStatus: document.querySelector("#model-status"),
  resultSection: document.querySelector("#result-section"),
  cropPreview: document.querySelector("#crop-preview"),
  predictedBoard: document.querySelector("#predicted-board"),
  boardFen: document.querySelector("#board-fen"),
  fullFenGroup: document.querySelector("#full-fen-group"),
  fullFen: document.querySelector("#full-fen"),
  meanConfidence: document.querySelector("#mean-confidence"),
  minimumConfidence: document.querySelector("#minimum-confidence"),
  lowConfidence: document.querySelector("#low-confidence"),
  confidenceNote: document.querySelector("#confidence-note"),
  predictionRows: document.querySelector("#prediction-rows"),
  timing: document.querySelector("#timing"),
  error: document.querySelector("#error"),
  correctionPanel: document.querySelector("#correction-panel"),
  correctionSummary: document.querySelector("#correction-summary"),
  correctionClass: document.querySelector("#correction-class"),
  applySquare: document.querySelector("#apply-square"),
  applyGroup: document.querySelector("#apply-group"),
};

const editorContext = elements.editor.getContext("2d");
const previewContext = elements.cropPreview.getContext("2d", { willReadFrequently: true });

const state = {
  image: null,
  crop: null,
  interaction: null,
  classifierSession: null,
  similaritySession: null,
  metadata: null,
  modelReady: false,
  rawPredictions: null,
  predictions: null,
  groups: [],
  fixedGroupLabels: new Map(),
  squareOverrides: new Map(),
  selectedSquare: null,
  elapsedMilliseconds: 0,
};

function setError(message = "") {
  elements.error.textContent = message;
  elements.error.hidden = message === "";
}

function setModelStatus(label, mode) {
  elements.modelStatus.textContent = label;
  elements.modelStatus.dataset.mode = mode;
}

async function loadModel() {
  try {
    setModelStatus("Loading model…", "loading");
    const metadataResponse = await fetch("./model/model.json");
    if (!metadataResponse.ok) throw new Error(`Model metadata returned ${metadataResponse.status}`);
    state.metadata = await metadataResponse.json();
    elements.correctionClass.replaceChildren();
    state.metadata.class_names.forEach((className, classId) => {
      const option = document.createElement("option");
      option.value = String(classId);
      option.textContent = className.replaceAll("_", " ");
      elements.correctionClass.append(option);
    });

    window.ort.env.wasm.numThreads = 1;
    window.ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    const options = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    [state.classifierSession, state.similaritySession] = await Promise.all([
      window.ort.InferenceSession.create(`./model/${state.metadata.model_path}`, options),
      window.ort.InferenceSession.create(
        `./model/${state.metadata.similarity.model_path}`,
        options,
      ),
    ]);
    state.modelReady = true;
    const modelBytes = state.metadata.model_bytes + state.metadata.similarity.model_bytes;
    setModelStatus(`Models ready · ${(modelBytes / 1_000_000).toFixed(1)} MB`, "ready");
    updateRecognizeButton();
  } catch (error) {
    console.error(error);
    setModelStatus("Model failed to load", "error");
    setError(`Could not load the browser model: ${error.message}`);
  }
}

function updateRecognizeButton() {
  elements.recognize.disabled = !(state.image && state.modelReady);
}

async function decodeImage(file) {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = "async";
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
      image.src = url;
    });
    return image;
  } catch (imageError) {
    if ("createImageBitmap" in window) {
      try {
        return await createImageBitmap(file, { imageOrientation: "from-image" });
      } catch (orientedBitmapError) {
        try {
          return await createImageBitmap(file);
        } catch (bitmapError) {
          console.debug(
            "Image element and ImageBitmap decoding failed",
            imageError,
            orientedBitmapError,
            bitmapError,
          );
        }
      }
    }
    throw new Error("This image format could not be opened. Try exporting it as a JPEG or PNG.");
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function acceptFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith("image/")) {
    setError("Choose an image from your camera, photo library, or files.");
    return;
  }
  try {
    setError();
    elements.uploadZone.querySelector("strong").textContent = "Opening image…";
    state.image = await decodeImage(file);
    state.rawPredictions = null;
    state.predictions = null;
    state.groups = [];
    state.fixedGroupLabels.clear();
    state.squareOverrides.clear();
    state.selectedSquare = null;
    elements.correctionPanel.hidden = true;
    resetCrop();
    elements.editorSection.hidden = false;
    elements.resultSection.hidden = true;
    elements.uploadZone.classList.add("has-image");
    elements.uploadZone.querySelector("strong").textContent = file.name;
    updateRecognizeButton();
    elements.editorSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setError(error.message);
    elements.uploadZone.querySelector("strong").textContent = "Drop a screenshot here";
  }
}

function resetCrop() {
  if (!state.image) return;
  const side = Math.min(state.image.width, state.image.height);
  state.crop = {
    x: (state.image.width - side) / 2,
    y: (state.image.height - side) / 2,
    size: side,
  };
  resizeEditor();
  drawEditor();
}

function resizeEditor() {
  if (!state.image) return;
  const maxWidth = 920;
  const maxHeight = 680;
  const scale = Math.min(maxWidth / state.image.width, maxHeight / state.image.height, 1);
  elements.editor.width = Math.max(1, Math.round(state.image.width * scale));
  elements.editor.height = Math.max(1, Math.round(state.image.height * scale));
  elements.editor.style.aspectRatio = `${elements.editor.width} / ${elements.editor.height}`;
}

function imageToCanvas(point) {
  return {
    x: point.x * (elements.editor.width / state.image.width),
    y: point.y * (elements.editor.height / state.image.height),
  };
}

function eventToImage(event) {
  const bounds = elements.editor.getBoundingClientRect();
  return {
    x: ((event.clientX - bounds.left) / bounds.width) * state.image.width,
    y: ((event.clientY - bounds.top) / bounds.height) * state.image.height,
  };
}

function cropCorners() {
  const { x, y, size } = state.crop;
  return {
    nw: { x, y },
    ne: { x: x + size, y },
    se: { x: x + size, y: y + size },
    sw: { x, y: y + size },
  };
}

function drawEditor() {
  if (!state.image || !state.crop) return;
  const width = elements.editor.width;
  const height = elements.editor.height;
  editorContext.clearRect(0, 0, width, height);
  editorContext.drawImage(state.image, 0, 0, width, height);
  editorContext.fillStyle = "rgba(4, 8, 18, 0.62)";
  editorContext.fillRect(0, 0, width, height);

  const topLeft = imageToCanvas(state.crop);
  const bottomRight = imageToCanvas({
    x: state.crop.x + state.crop.size,
    y: state.crop.y + state.crop.size,
  });
  const cropWidth = bottomRight.x - topLeft.x;
  const cropHeight = bottomRight.y - topLeft.y;
  editorContext.drawImage(
    state.image,
    state.crop.x,
    state.crop.y,
    state.crop.size,
    state.crop.size,
    topLeft.x,
    topLeft.y,
    cropWidth,
    cropHeight,
  );
  editorContext.strokeStyle = "#f6c453";
  editorContext.lineWidth = 3;
  editorContext.strokeRect(topLeft.x, topLeft.y, cropWidth, cropHeight);
  editorContext.strokeStyle = "rgba(246, 196, 83, 0.5)";
  editorContext.lineWidth = 1;
  for (let index = 1; index < 8; index += 1) {
    const offsetX = (cropWidth * index) / 8;
    const offsetY = (cropHeight * index) / 8;
    editorContext.beginPath();
    editorContext.moveTo(topLeft.x + offsetX, topLeft.y);
    editorContext.lineTo(topLeft.x + offsetX, topLeft.y + cropHeight);
    editorContext.moveTo(topLeft.x, topLeft.y + offsetY);
    editorContext.lineTo(topLeft.x + cropWidth, topLeft.y + offsetY);
    editorContext.stroke();
  }
  editorContext.fillStyle = "#fff7db";
  editorContext.strokeStyle = "#4b3510";
  for (const corner of Object.values(cropCorners())) {
    const canvasCorner = imageToCanvas(corner);
    editorContext.beginPath();
    editorContext.arc(canvasCorner.x, canvasCorner.y, 7, 0, Math.PI * 2);
    editorContext.fill();
    editorContext.stroke();
  }
  elements.cropSize.textContent = `${Math.round(state.crop.size)} × ${Math.round(state.crop.size)} px`;
}

function hitTest(point) {
  const canvasScale = elements.editor.width / state.image.width;
  const handleRadius = 18 / canvasScale;
  for (const [name, corner] of Object.entries(cropCorners())) {
    if (Math.hypot(point.x - corner.x, point.y - corner.y) <= handleRadius) {
      return { type: "resize", corner: name };
    }
  }
  const { x, y, size } = state.crop;
  if (point.x >= x && point.x <= x + size && point.y >= y && point.y <= y + size) {
    return { type: "move" };
  }
  return null;
}

function beginInteraction(event) {
  if (!state.crop) return;
  const point = eventToImage(event);
  const hit = hitTest(point);
  if (!hit) return;
  elements.editor.setPointerCapture(event.pointerId);
  state.interaction = { ...hit, start: point, original: { ...state.crop } };
  if (hit.type === "resize") {
    const opposite = { nw: "se", ne: "sw", se: "nw", sw: "ne" }[hit.corner];
    state.interaction.anchor = cropCorners()[opposite];
  }
  event.preventDefault();
}

function moveInteraction(event) {
  if (!state.interaction) return;
  const point = eventToImage(event);
  if (state.interaction.type === "move") {
    const dx = point.x - state.interaction.start.x;
    const dy = point.y - state.interaction.start.y;
    const { size } = state.interaction.original;
    state.crop.x = Math.min(Math.max(0, state.interaction.original.x + dx), state.image.width - size);
    state.crop.y = Math.min(Math.max(0, state.interaction.original.y + dy), state.image.height - size);
  } else {
    const { anchor, corner } = state.interaction;
    const signX = corner.includes("w") ? -1 : 1;
    const signY = corner.includes("n") ? -1 : 1;
    const maxX = signX < 0 ? anchor.x : state.image.width - anchor.x;
    const maxY = signY < 0 ? anchor.y : state.image.height - anchor.y;
    const maxSide = Math.min(maxX, maxY);
    const requested = Math.max(Math.abs(point.x - anchor.x), Math.abs(point.y - anchor.y));
    const minimum = Math.min(64, maxSide);
    const size = Math.min(Math.max(requested, minimum), maxSide);
    state.crop = {
      x: signX < 0 ? anchor.x - size : anchor.x,
      y: signY < 0 ? anchor.y - size : anchor.y,
      size,
    };
  }
  drawEditor();
  event.preventDefault();
}

function endInteraction(event) {
  if (!state.interaction) return;
  state.interaction = null;
  if (elements.editor.hasPointerCapture(event.pointerId)) {
    elements.editor.releasePointerCapture(event.pointerId);
  }
}

function prepareInputTensor() {
  const { x, y, size } = state.crop;
  previewContext.imageSmoothingEnabled = true;
  previewContext.imageSmoothingQuality = "high";
  previewContext.clearRect(0, 0, BOARD_SIZE, BOARD_SIZE);
  previewContext.drawImage(state.image, x, y, size, size, 0, 0, BOARD_SIZE, BOARD_SIZE);
  const pixels = previewContext.getImageData(0, 0, BOARD_SIZE, BOARD_SIZE).data;
  const plane = SQUARE_SIZE * SQUARE_SIZE;
  const values = new Float32Array(NUM_SQUARES * 3 * plane);

  for (let square = 0; square < NUM_SQUARES; square += 1) {
    const squareRow = Math.floor(square / 8);
    const squareColumn = square % 8;
    const batchOffset = square * 3 * plane;
    for (let row = 0; row < SQUARE_SIZE; row += 1) {
      for (let column = 0; column < SQUARE_SIZE; column += 1) {
        const sourceX = squareColumn * SQUARE_SIZE + column;
        const sourceY = squareRow * SQUARE_SIZE + row;
        const pixelOffset = (sourceY * BOARD_SIZE + sourceX) * 4;
        const planeOffset = row * SQUARE_SIZE + column;
        values[batchOffset + planeOffset] = pixels[pixelOffset] / 127.5 - 1;
        values[batchOffset + plane + planeOffset] = pixels[pixelOffset + 1] / 127.5 - 1;
        values[batchOffset + 2 * plane + planeOffset] = pixels[pixelOffset + 2] / 127.5 - 1;
      }
    }
  }
  return new window.ort.Tensor("float32", values, [NUM_SQUARES, 3, SQUARE_SIZE, SQUARE_SIZE]);
}

function readPredictions(logits) {
  const classCount = state.metadata.class_names.length;
  if (logits.dims[0] !== NUM_SQUARES || logits.dims[1] !== classCount) {
    throw new Error(`Unexpected model output shape: ${logits.dims.join(" × ")}`);
  }
  const whiteAtBottom = elements.orientation.value === "white";
  const predictions = new Array(NUM_SQUARES);
  for (let rasterIndex = 0; rasterIndex < NUM_SQUARES; rasterIndex += 1) {
    const start = rasterIndex * classCount;
    const rawLogits = Array.from(logits.data.slice(start, start + classCount));
    const result = bestClass(rawLogits);
    const fenIndex = whiteAtBottom ? rasterIndex : NUM_SQUARES - 1 - rasterIndex;
    predictions[fenIndex] = {
      ...result,
      rawClassId: result.classId,
      logits: rawLogits,
      square: squareName(fenIndex),
      groupId: null,
    };
  }
  return predictions;
}

function readEmbeddings(tensor) {
  const embeddingSize = state.metadata.similarity.embedding_size;
  if (tensor.dims[0] !== NUM_SQUARES || tensor.dims[1] !== embeddingSize) {
    throw new Error(`Unexpected embedding shape: ${tensor.dims.join(" × ")}`);
  }
  const whiteAtBottom = elements.orientation.value === "white";
  const embeddings = new Array(NUM_SQUARES);
  for (let rasterIndex = 0; rasterIndex < NUM_SQUARES; rasterIndex += 1) {
    const fenIndex = whiteAtBottom ? rasterIndex : NUM_SQUARES - 1 - rasterIndex;
    const start = rasterIndex * embeddingSize;
    embeddings[fenIndex] = Array.from(tensor.data.slice(start, start + embeddingSize));
  }
  return embeddings;
}

function buildGroups(rawPredictions, embeddings) {
  const candidates = rawPredictions.map((_, index) => index);
  const embeddingSize = state.metadata.similarity.embedding_size;
  const compact = new Float32Array(candidates.length * embeddingSize);
  candidates.forEach((squareIndex, row) => {
    compact.set(embeddings[squareIndex], row * embeddingSize);
  });
  const similarities = cosineSimilarityMatrix(compact, candidates.length, embeddingSize);
  return completeLinkageClusters(
    similarities,
    candidates,
    state.metadata.similarity.similarity_threshold,
  );
}

function applyGroupedAssignments() {
  const classCount = state.metadata.class_names.length;
  const squareLogits = state.rawPredictions.map((prediction) => prediction.logits);
  const assignments = assignGroupLabels(
    squareLogits,
    state.groups,
    classCount,
    state.metadata.similarity.duplicate_penalty,
    state.fixedGroupLabels,
  );
  const assignmentByGroup = new Map(
    assignments.map((assignment) => [assignment.groupId, assignment]),
  );
  const predictions = state.rawPredictions.map((prediction) => ({ ...prediction }));
  for (const group of state.groups) {
    const assignment = assignmentByGroup.get(group.groupId);
    group.classId = assignment.classId;
    group.assignmentConfidence = assignment.confidence;
    for (const squareIndex of group.squareIndices) {
      predictions[squareIndex].classId = assignment.classId;
      predictions[squareIndex].confidence = assignment.confidence;
      predictions[squareIndex].probabilities = assignment.probabilities;
      predictions[squareIndex].groupId = group.groupId;
    }
  }
  for (const [squareIndex, classId] of state.squareOverrides) {
    predictions[squareIndex].classId = classId;
    predictions[squareIndex].confidence = 1;
  }
  state.predictions = predictions;
}

function renderBoard(predictions, threshold) {
  elements.predictedBoard.replaceChildren();
  const whiteAtBottom = elements.orientation.value === "white";
  for (let displayIndex = 0; displayIndex < NUM_SQUARES; displayIndex += 1) {
    const fenIndex = whiteAtBottom ? displayIndex : NUM_SQUARES - 1 - displayIndex;
    const prediction = predictions[fenIndex];
    const cell = document.createElement("div");
    const row = Math.floor(displayIndex / 8);
    const column = displayIndex % 8;
    cell.className = `board-square ${(row + column) % 2 === 0 ? "light" : "dark"}`;
    cell.dataset.squareIndex = String(fenIndex);
    if (prediction.confidence < threshold) cell.classList.add("uncertain");
    if (state.selectedSquare !== null) {
      const selected = predictions[state.selectedSquare];
      if (fenIndex === state.selectedSquare) cell.classList.add("selected-square");
      if (selected.groupId !== null && prediction.groupId === selected.groupId) {
        cell.classList.add("selected-group");
      }
    }
    const symbol = state.metadata.fen_symbols[prediction.classId];
    if (symbol) cell.classList.add(symbol === symbol.toUpperCase() ? "white-piece" : "black-piece");
    cell.textContent = PIECE_GLYPHS[symbol] ?? "";
    cell.title = `${prediction.square}: ${state.metadata.class_names[prediction.classId]} (${(prediction.confidence * 100).toFixed(1)}%)`;
    cell.addEventListener("click", () => selectSquare(fenIndex));
    elements.predictedBoard.append(cell);
  }
}

function selectSquare(squareIndex) {
  if (!state.predictions) return;
  state.selectedSquare = squareIndex;
  const prediction = state.predictions[squareIndex];
  const group = state.groups.find((candidate) => candidate.groupId === prediction.groupId);
  elements.correctionClass.value = String(prediction.classId);
  elements.correctionSummary.textContent = group
    ? `${prediction.square} belongs to group ${group.groupId}: ${group.squareIndices.map(squareName).join(", ")}`
    : `${prediction.square} is not in an appearance group.`;
  elements.applyGroup.disabled = !group;
  elements.correctionPanel.hidden = false;
  renderBoard(state.predictions, Number(elements.threshold.value));
}

function renderPredictionTable(predictions, threshold) {
  elements.predictionRows.replaceChildren();
  for (const prediction of predictions) {
    const row = document.createElement("tr");
    if (prediction.confidence < threshold) row.classList.add("low-row");
    const values = [
      prediction.square,
      state.metadata.class_names[prediction.classId].replaceAll("_", " "),
      state.metadata.fen_symbols[prediction.classId] || "empty",
      `${(prediction.confidence * 100).toFixed(1)}%`,
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    elements.predictionRows.append(row);
  }
}

function showResults(predictions, elapsedMilliseconds) {
  const classIds = predictions.map((prediction) => prediction.classId);
  const boardFen = buildBoardFen(classIds, state.metadata.fen_symbols);
  const confidences = predictions.map((prediction) => prediction.confidence);
  const threshold = Number(elements.threshold.value);
  const lowConfidence = predictions
    .filter((prediction) => prediction.confidence < threshold)
    .sort((left, right) => left.confidence - right.confidence);

  elements.boardFen.textContent = boardFen;
  if (elements.sideToMove.value === "none") {
    elements.fullFenGroup.hidden = true;
  } else {
    elements.fullFenGroup.hidden = false;
    elements.fullFen.textContent = `${boardFen} ${elements.sideToMove.value} - - 0 1`;
  }
  elements.meanConfidence.textContent = `${(confidences.reduce((sum, value) => sum + value, 0) / 64 * 100).toFixed(1)}%`;
  elements.minimumConfidence.textContent = `${(Math.min(...confidences) * 100).toFixed(1)}%`;
  elements.lowConfidence.textContent = String(lowConfidence.length);
  elements.confidenceNote.textContent = lowConfidence.length
    ? `Check these squares first: ${lowConfidence.slice(0, 12).map((prediction) => prediction.square).join(", ")}`
    : "Every square is above the selected confidence threshold.";
  elements.timing.textContent = `Classified 64 squares locally in ${elapsedMilliseconds.toFixed(0)} ms.`;
  renderBoard(predictions, threshold);
  renderPredictionTable(predictions, threshold);
  elements.resultSection.hidden = false;
  elements.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function recognize() {
  if (!state.image || !state.classifierSession || !state.similaritySession) return;
  setError();
  elements.recognize.disabled = true;
  elements.recognize.textContent = "Recognizing…";
  await new Promise((resolve) => requestAnimationFrame(resolve));
  try {
    const tensor = prepareInputTensor();
    const start = performance.now();
    // ONNX Runtime Web's single WASM worker cannot execute two sessions at
    // once. Run them sequentially while reusing the same immutable input.
    const classifierOutputs = await state.classifierSession.run({
      [state.metadata.input_name]: tensor,
    });
    const similarityOutputs = await state.similaritySession.run({
      [state.metadata.similarity.input_name]: tensor,
    });
    const elapsed = performance.now() - start;
    state.rawPredictions = readPredictions(classifierOutputs[state.metadata.output_name]);
    const embeddings = readEmbeddings(
      similarityOutputs[state.metadata.similarity.output_name],
    );
    state.groups = buildGroups(state.rawPredictions, embeddings);
    state.fixedGroupLabels.clear();
    state.squareOverrides.clear();
    state.selectedSquare = null;
    state.elapsedMilliseconds = elapsed;
    applyGroupedAssignments();
    elements.correctionPanel.hidden = true;
    showResults(state.predictions, elapsed);
  } catch (error) {
    console.error(error);
    setError(`Recognition failed: ${error.message}`);
  } finally {
    elements.recognize.textContent = "Recognize position";
    updateRecognizeButton();
  }
}

elements.upload.addEventListener("change", async (event) => {
  const file = event.currentTarget.files?.[0];
  event.currentTarget.value = "";
  await acceptFile(file);
});
elements.uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.uploadZone.classList.add("dragging");
});
elements.uploadZone.addEventListener("dragleave", () => elements.uploadZone.classList.remove("dragging"));
elements.uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.uploadZone.classList.remove("dragging");
  acceptFile(event.dataTransfer.files[0]);
});
elements.resetCrop.addEventListener("click", resetCrop);
elements.editor.addEventListener("pointerdown", beginInteraction);
elements.editor.addEventListener("pointermove", moveInteraction);
elements.editor.addEventListener("pointerup", endInteraction);
elements.editor.addEventListener("pointercancel", endInteraction);
elements.threshold.addEventListener("input", () => {
  elements.thresholdValue.textContent = `${Math.round(Number(elements.threshold.value) * 100)}%`;
});
elements.recognize.addEventListener("click", recognize);
elements.applySquare.addEventListener("click", () => {
  if (state.selectedSquare === null) return;
  state.squareOverrides.set(state.selectedSquare, Number(elements.correctionClass.value));
  applyGroupedAssignments();
  showResults(state.predictions, state.elapsedMilliseconds);
  selectSquare(state.selectedSquare);
});
elements.applyGroup.addEventListener("click", () => {
  if (state.selectedSquare === null) return;
  const groupId = state.predictions[state.selectedSquare].groupId;
  if (groupId === null) return;
  state.fixedGroupLabels.set(groupId, Number(elements.correctionClass.value));
  applyGroupedAssignments();
  showResults(state.predictions, state.elapsedMilliseconds);
  selectSquare(state.selectedSquare);
});
document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.querySelector(`#${button.dataset.copy}`);
    await navigator.clipboard.writeText(target.textContent);
    const original = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => { button.textContent = original; }, 1200);
  });
});

loadModel();
