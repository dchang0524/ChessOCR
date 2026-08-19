import assert from "node:assert/strict";
import test from "node:test";

import { bestClass, buildBoardFen, softmax, squareName } from "../web/chess.js";

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
