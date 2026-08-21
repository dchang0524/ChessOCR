import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";

import { bestClass, buildBoardFen, softmax, squareName } from "../web/chess.js";
import {
  assignGroupLabels,
  completeLinkageClusters,
  cosineSimilarityMatrix,
} from "../web/grouping.js";

test("browser catalog exposes the three requested model modes with valid assets", async () => {
  const modelDirectory = new URL("../web/model/", import.meta.url);
  const catalog = JSON.parse(await readFile(new URL("models.json", modelDirectory), "utf8"));
  const expectedIds = [
    "kaggle-baseline",
    "dino-kaggle-grouped",
    "dino-generated-grouped",
  ];
  assert.deepEqual(catalog.models.map((model) => model.id), expectedIds);
  assert.equal(catalog.default_model, "kaggle-baseline");

  for (const definition of catalog.models) {
    const metadata = JSON.parse(
      await readFile(new URL(definition.metadata_path, modelDirectory), "utf8"),
    );
    assert.equal(Boolean(metadata.similarity), definition.grouping);
    const modelPath = new URL(metadata.model_path, modelDirectory);
    const modelBytes = await readFile(modelPath);
    const modelStat = await stat(modelPath);
    assert.equal(modelStat.size, metadata.model_bytes);
    assert.ok(modelStat.size <= 25 * 1024 * 1024);
    assert.equal(createHash("sha256").update(modelBytes).digest("hex"), metadata.model_sha256);
  }
});

test("buildBoardFen encodes the starting position", () => {
  const classIds = [
    10, 8, 9, 11, 12, 9, 8, 10,
    ...Array(8).fill(7),
    ...Array(32).fill(0),
    ...Array(8).fill(1),
    4, 2, 3, 5, 6, 3, 2, 4,
  ];
  assert.equal(buildBoardFen(classIds), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR");
});

test("squareName uses FEN order", () => {
  assert.equal(squareName(0), "a8");
  assert.equal(squareName(63), "h1");
  assert.throws(() => squareName(64), RangeError);
});

test("softmax is stable and bestClass returns the maximum", () => {
  const probabilities = softmax([1000, 1001, 999]);
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  const prediction = bestClass([0, -2, 4, 1]);
  assert.equal(prediction.classId, 2);
  assert.equal(prediction.confidence, Math.max(...prediction.probabilities));
});

test("complete linkage groups matching embeddings", () => {
  const embeddings = new Float32Array([1, 0, 0.99, 0.05, 0, 1]);
  const similarities = cosineSimilarityMatrix(embeddings, 3, 2);
  const groups = completeLinkageClusters(similarities, [2, 7, 11], 0.9);
  assert.deepEqual(groups.map((group) => group.squareIndices), [[2, 7], [11]]);
});

test("cross-background threshold only relaxes opposite-colour square pairs", () => {
  const similarities = [
    [1, 0.95, 0.95],
    [0.95, 1, 0.2],
    [0.95, 0.2, 1],
  ];
  // a8 (0) and b8 (1) have opposite backgrounds; a8 and c8 (2) do not.
  const groups = completeLinkageClusters(similarities, [0, 1, 2], 0.98, 0.94);
  assert.deepEqual(groups.map((group) => group.squareIndices), [[0, 1], [2]]);
});

test("joint assignment gives a weaker group its second choice", () => {
  const logits = Array.from({ length: 3 }, () => Array(13).fill(-10));
  logits[0][1] = 4;
  logits[0][3] = 1;
  logits[1][1] = 3;
  logits[1][3] = 2.8;
  logits[2][1] = 3;
  logits[2][3] = 2.8;
  const groups = [
    { groupId: 0, squareIndices: [0], confidence: 1 },
    { groupId: 1, squareIndices: [1, 2], confidence: 1 },
  ];
  const assignments = assignGroupLabels(logits, groups, 13, 2);
  assert.deepEqual(assignments.map((assignment) => assignment.classId), [1, 3]);
});

test("empty is a valid repeatable group label", () => {
  const logits = Array.from({ length: 4 }, () => Array(13).fill(-10));
  for (const squareLogits of logits) squareLogits[0] = 10;
  const groups = [
    { groupId: 0, squareIndices: [0, 1], confidence: 1 },
    { groupId: 1, squareIndices: [2, 3], confidence: 1 },
  ];
  const assignments = assignGroupLabels(logits, groups, 13);
  assert.deepEqual(assignments.map((assignment) => assignment.classId), [0, 0]);
});
