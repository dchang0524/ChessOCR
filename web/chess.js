export const FEN_SYMBOLS = ["", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"];

export const PIECE_GLYPHS = {
  P: "♙",
  N: "♘",
  B: "♗",
  R: "♖",
  Q: "♕",
  K: "♔",
  p: "♟",
  n: "♞",
  b: "♝",
  r: "♜",
  q: "♛",
  k: "♚",
};

export function squareName(index) {
  if (!Number.isInteger(index) || index < 0 || index >= 64) {
    throw new RangeError(`Square index must be in [0, 63], got ${index}`);
  }
  return `${"abcdefgh"[index % 8]}${8 - Math.floor(index / 8)}`;
}

export function buildBoardFen(classIds, fenSymbols = FEN_SYMBOLS) {
  if (classIds.length !== 64) {
    throw new Error(`Expected 64 class IDs, got ${classIds.length}`);
  }
  const ranks = [];
  for (let rank = 0; rank < 8; rank += 1) {
    let encoded = "";
    let emptyRun = 0;
    for (let file = 0; file < 8; file += 1) {
      const classId = classIds[rank * 8 + file];
      if (!Number.isInteger(classId) || classId < 0 || classId >= fenSymbols.length) {
        throw new Error(`Invalid class ID ${classId} at square ${rank * 8 + file}`);
      }
      const symbol = fenSymbols[classId];
      if (symbol === "") {
        emptyRun += 1;
      } else {
        if (emptyRun > 0) encoded += String(emptyRun);
        emptyRun = 0;
        encoded += symbol;
      }
    }
    if (emptyRun > 0) encoded += String(emptyRun);
    ranks.push(encoded);
  }
  return ranks.join("/");
}

export function softmax(values) {
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

export function bestClass(logits) {
  const probabilities = softmax(logits);
  let classId = 0;
  for (let index = 1; index < probabilities.length; index += 1) {
    if (probabilities[index] > probabilities[classId]) classId = index;
  }
  return { classId, confidence: probabilities[classId], probabilities };
}
