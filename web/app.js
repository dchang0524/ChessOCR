import { PIECE_GLYPHS, bestClass, buildBoardFen, squareName } from "./chess.js";
import {
  assignGroupLabels,
  completeLinkageClusters,
  cosineSimilarityMatrix,
} from "./grouping.js";

const ORT_VERSION = "1.22.0";
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
  modelSelect: document.querySelector("#model-select"),
  modelDescription: document.querySelector("#model-description"),
  threshold: document.querySelector("#threshold"),
  thresholdValue: document.querySelector("#threshold-value"),
  recognize: document.querySelector("#recognize"),
  modelStatus: document.querySelector("#model-status"),
  resultSection: document.querySelector("#result-section"),
  cropPreview: document.querySelector("#crop-preview"),
  predictedBoard: document.querySelector("#predicted-board"),
  groupBrowser: document.querySelector("#group-browser"),
  groupList: document.querySelector("#group-list"),
  groupSummary: document.querySelector("#group-summary"),
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
  correctionHelp: document.querySelector("#correction-help"),
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
  modelCatalog: null,
  activeModel: null,
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

function groupingEnabled() {
  return Boolean(state.activeModel?.grouping && state.metadata?.similarity);
}

function clearInferenceResults() {
  state.rawPredictions = null;
  state.predictions = null;
  state.groups = [];
  state.fixedGroupLabels.clear();
  state.squareOverrides.clear();
  state.selectedSquare = null;
  elements.correctionPanel.hidden = true;
  elements.resultSection.hidden = true;
}

function releaseCurrentSessions() {
  const classifier = state.classifierSession;
  const similarity = state.similaritySession;
  state.classifierSession = null;
  state.similaritySession = null;
  if (similarity && similarity !== classifier) similarity.release();
  if (classifier) classifier.release();
}

async function loadModel(modelId) {
  const modelDefinition = state.modelCatalog.models.find((model) => model.id === modelId);
  if (!modelDefinition) throw new Error(`Unknown model choice: ${modelId}`);
  try {
    setError();
    state.modelReady = false;
    state.activeModel = modelDefinition;
    elements.modelSelect.disabled = true;
    elements.modelDescription.textContent = modelDefinition.description;
    setModelStatus(`Loading ${modelDefinition.short_label}…`, "loading");
    updateRecognizeButton();
    clearInferenceResults();
    releaseCurrentSessions();

    const metadataResponse = await fetch(
      `./model/${modelDefinition.metadata_path}`,
      { cache: "no-store" },
    );
    if (!metadataResponse.ok) throw new Error(`Model metadata returned ${metadataResponse.status}`);
    state.metadata = await metadataResponse.json();
    if (modelDefinition.grouping !== Boolean(state.metadata.similarity)) {
      throw new Error("Model catalog grouping mode does not match its metadata");
    }
    const modelBoardSize = state.metadata.input_size * 8;
    elements.cropPreview.width = modelBoardSize;
    elements.cropPreview.height = modelBoardSize;
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
    const classifierSession = await window.ort.InferenceSession.create(
      `./model/${state.metadata.model_path}?v=${state.metadata.model_sha256}`,
      options,
    );
    let similaritySession = null;
    if (state.metadata.similarity?.model_path === state.metadata.model_path) {
      similaritySession = classifierSession;
    } else if (state.metadata.similarity) {
      similaritySession = await window.ort.InferenceSession.create(
        `./model/${state.metadata.similarity.model_path}?v=${state.metadata.similarity.model_sha256}`,
        options,
      );
    }
    state.classifierSession = classifierSession;
    state.similaritySession = similaritySession;
    state.modelReady = true;
    const modelBytes = !state.similaritySession || state.classifierSession === state.similaritySession
      ? state.metadata.model_bytes
      : state.metadata.model_bytes + state.metadata.similarity.model_bytes;
    const mode = groupingEnabled() ? "grouped" : "one-shot";
    setModelStatus(
      `${modelDefinition.short_label} ready · ${mode} · ${(modelBytes / 1_000_000).toFixed(1)} MB`,
      "ready",
    );
    updateRecognizeButton();
  } catch (error) {
    console.error(error);
    releaseCurrentSessions();
    state.modelReady = false;
    setModelStatus("Model failed to load", "error");
    setError(`Could not load the browser model: ${error.message}`);
  } finally {
    elements.modelSelect.disabled = false;
  }
}

async function loadModelCatalog() {
  try {
    setModelStatus("Loading model choices…", "loading");
    const response = await fetch("./model/models.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Model catalog returned ${response.status}`);
    const catalog = await response.json();
    if (!Array.isArray(catalog.models) || catalog.models.length === 0) {
      throw new Error("Model catalog is empty");
    }
    if (!catalog.models.some((model) => model.id === catalog.default_model)) {
      throw new Error("Default model is missing from the catalog");
    }
    state.modelCatalog = catalog;
    elements.modelSelect.replaceChildren();
    for (const model of catalog.models) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.label;
      elements.modelSelect.append(option);
    }
    elements.modelSelect.value = catalog.default_model;
    await loadModel(catalog.default_model);
  } catch (error) {
    console.error(error);
    state.modelReady = false;
    setModelStatus("Models failed to load", "error");
    setError(`Could not load the model choices: ${error.message}`);
    elements.modelSelect.disabled = true;
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
    clearInferenceResults();
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
  const squareSize = state.metadata.input_size;
  const boardSize = squareSize * 8;
  const { x, y, size } = state.crop;
  previewContext.imageSmoothingEnabled = true;
  previewContext.imageSmoothingQuality = "high";
  previewContext.clearRect(0, 0, boardSize, boardSize);
  previewContext.drawImage(state.image, x, y, size, size, 0, 0, boardSize, boardSize);
  const pixels = previewContext.getImageData(0, 0, boardSize, boardSize).data;
  const plane = squareSize * squareSize;
  const values = new Float32Array(NUM_SQUARES * 3 * plane);

  for (let square = 0; square < NUM_SQUARES; square += 1) {
    const squareRow = Math.floor(square / 8);
    const squareColumn = square % 8;
    const batchOffset = square * 3 * plane;
    for (let row = 0; row < squareSize; row += 1) {
      for (let column = 0; column < squareSize; column += 1) {
        const sourceX = squareColumn * squareSize + column;
        const sourceY = squareRow * squareSize + row;
        const pixelOffset = (sourceY * boardSize + sourceX) * 4;
        const planeOffset = row * squareSize + column;
        values[batchOffset + planeOffset] = pixels[pixelOffset] / 127.5 - 1;
        values[batchOffset + plane + planeOffset] = pixels[pixelOffset + 1] / 127.5 - 1;
        values[batchOffset + 2 * plane + planeOffset] = pixels[pixelOffset + 2] / 127.5 - 1;
      }
    }
  }
  return new window.ort.Tensor("float32", values, [NUM_SQUARES, 3, squareSize, squareSize]);
}

async function runJointModelInBatches(tensor) {
  const batchSize = state.metadata.inference_batch_size ?? NUM_SQUARES;
  const valuesPerSquare = tensor.data.length / NUM_SQUARES;
  const logits = new Float32Array(NUM_SQUARES * state.metadata.class_names.length);
  const embeddings = new Float32Array(
    NUM_SQUARES * state.metadata.similarity.embedding_size,
  );
  for (let start = 0; start < NUM_SQUARES; start += batchSize) {
    const count = Math.min(batchSize, NUM_SQUARES - start);
    const batch = new window.ort.Tensor(
      "float32",
      tensor.data.slice(start * valuesPerSquare, (start + count) * valuesPerSquare),
      [count, 3, state.metadata.input_size, state.metadata.input_size],
    );
    const outputs = await state.classifierSession.run({
      [state.metadata.input_name]: batch,
    });
    logits.set(outputs[state.metadata.output_name].data, start * state.metadata.class_names.length);
    embeddings.set(
      outputs[state.metadata.similarity.output_name].data,
      start * state.metadata.similarity.embedding_size,
    );
  }
  return {
    logits: { data: logits, dims: [NUM_SQUARES, state.metadata.class_names.length] },
    embeddings: {
      data: embeddings,
      dims: [NUM_SQUARES, state.metadata.similarity.embedding_size],
    },
  };
}

async function runClassifierInBatches(tensor) {
  const batchSize = state.metadata.inference_batch_size ?? NUM_SQUARES;
  const valuesPerSquare = tensor.data.length / NUM_SQUARES;
  const logits = new Float32Array(NUM_SQUARES * state.metadata.class_names.length);
  for (let start = 0; start < NUM_SQUARES; start += batchSize) {
    const count = Math.min(batchSize, NUM_SQUARES - start);
    const batch = new window.ort.Tensor(
      "float32",
      tensor.data.slice(start * valuesPerSquare, (start + count) * valuesPerSquare),
      [count, 3, state.metadata.input_size, state.metadata.input_size],
    );
    const outputs = await state.classifierSession.run({
      [state.metadata.input_name]: batch,
    });
    logits.set(outputs[state.metadata.output_name].data, start * state.metadata.class_names.length);
  }
  return { data: logits, dims: [NUM_SQUARES, state.metadata.class_names.length] };
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
    state.metadata.similarity.cross_background_similarity_threshold ?? null,
  );
}

function applyGroupedAssignments() {
  if (!groupingEnabled()) {
    const predictions = state.rawPredictions.map((prediction) => ({ ...prediction }));
    for (const [squareIndex, classId] of state.squareOverrides) {
      predictions[squareIndex].classId = classId;
      predictions[squareIndex].confidence = 1;
    }
    state.predictions = predictions;
    return;
  }
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

function selectedGroupId() {
  if (state.selectedSquare === null || !state.predictions) return null;
  return state.predictions[state.selectedSquare].groupId;
}

function renderGroups() {
  elements.groupList.replaceChildren();
  elements.groupBrowser.hidden = !groupingEnabled();
  if (!groupingEnabled()) return;
  const activeGroupId = selectedGroupId();
  elements.groupSummary.textContent = `${state.groups.length} groups found · select one to highlight its squares.`;
  for (const group of state.groups) {
    const className = state.metadata.class_names[group.classId];
    const symbol = state.metadata.fen_symbols[group.classId];
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "group-tile";
    tile.dataset.groupId = String(group.groupId);
    tile.setAttribute("aria-pressed", String(group.groupId === activeGroupId));
    if (group.groupId === activeGroupId) tile.classList.add("active");
    tile.title = `Group ${group.groupId}: ${className.replaceAll("_", " ")} · ${group.squareIndices.map(squareName).join(", ")}`;

    const icon = document.createElement("span");
    icon.className = "group-icon";
    if (symbol) {
      icon.textContent = PIECE_GLYPHS[symbol] ?? symbol;
      icon.classList.add(symbol === symbol.toUpperCase() ? "white-piece" : "black-piece");
    } else {
      icon.textContent = "□";
      icon.classList.add("empty-group");
    }
    icon.setAttribute("aria-hidden", "true");

    const details = document.createElement("span");
    details.className = "group-details";
    const label = document.createElement("strong");
    label.textContent = className.replaceAll("_", " ");
    const metadata = document.createElement("span");
    const memberLabel = group.squareIndices.length === 1 ? "square" : "squares";
    metadata.textContent = `Group ${group.groupId} · ${group.squareIndices.length} ${memberLabel} · ${(group.assignmentConfidence * 100).toFixed(0)}%`;
    details.append(label, metadata);
    tile.append(icon, details);
    tile.addEventListener("click", () => selectGroup(group.groupId));
    elements.groupList.append(tile);
  }
}

function selectGroup(groupId) {
  const group = state.groups.find((candidate) => candidate.groupId === groupId);
  if (!group || group.squareIndices.length === 0) return;
  selectSquare(group.squareIndices[0]);
}

function selectSquare(squareIndex) {
  if (!state.predictions) return;
  state.selectedSquare = squareIndex;
  const prediction = state.predictions[squareIndex];
  const group = state.groups.find((candidate) => candidate.groupId === prediction.groupId);
  elements.correctionClass.value = String(prediction.classId);
  elements.correctionSummary.textContent = group
    ? `${prediction.square} belongs to group ${group.groupId}: ${group.squareIndices.map(squareName).join(", ")}`
    : `${prediction.square} was classified independently by the one-shot model.`;
  elements.correctionHelp.textContent = group
    ? "Apply a correction to this square or its whole appearance group. Fixed group labels are preserved while the remaining groups are reassigned."
    : "The one-shot model has no appearance groups, so corrections apply to one square at a time.";
  elements.applyGroup.disabled = !group;
  elements.correctionPanel.hidden = false;
  renderBoard(state.predictions, Number(elements.threshold.value));
  renderGroups();
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
  renderGroups();
  renderPredictionTable(predictions, threshold);
  elements.resultSection.hidden = false;
  elements.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function recognize() {
  if (!state.image || !state.classifierSession) return;
  setError();
  elements.recognize.disabled = true;
  elements.modelSelect.disabled = true;
  elements.recognize.textContent = "Recognizing…";
  await new Promise((resolve) => requestAnimationFrame(resolve));
  try {
    const tensor = prepareInputTensor();
    const start = performance.now();
    let logits;
    let embeddingTensor;
    if (!groupingEnabled()) {
      logits = await runClassifierInBatches(tensor);
    } else if (state.classifierSession === state.similaritySession) {
      const jointOutputs = await runJointModelInBatches(tensor);
      logits = jointOutputs.logits;
      embeddingTensor = jointOutputs.embeddings;
    } else {
      // ONNX Runtime Web's single WASM worker cannot execute two sessions at
      // once. Run them sequentially while reusing the same immutable input.
      const classifierOutputs = await state.classifierSession.run({
        [state.metadata.input_name]: tensor,
      });
      const similarityOutputs = await state.similaritySession.run({
        [state.metadata.similarity.input_name]: tensor,
      });
      logits = classifierOutputs[state.metadata.output_name];
      embeddingTensor = similarityOutputs[state.metadata.similarity.output_name];
    }
    const elapsed = performance.now() - start;
    state.rawPredictions = readPredictions(logits);
    if (groupingEnabled()) {
      const embeddings = readEmbeddings(embeddingTensor);
      state.groups = buildGroups(state.rawPredictions, embeddings);
    } else {
      state.groups = [];
    }
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
    elements.modelSelect.disabled = !state.modelCatalog;
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
elements.modelSelect.addEventListener("change", async () => {
  await loadModel(elements.modelSelect.value);
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

loadModelCatalog();
