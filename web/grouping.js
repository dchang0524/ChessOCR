export function cosineSimilarityMatrix(flatEmbeddings, count, embeddingSize) {
  const matrix = Array.from({ length: count }, () => Array(count).fill(0));
  for (let left = 0; left < count; left += 1) {
    for (let right = left; right < count; right += 1) {
      let dot = 0;
      let leftNorm = 0;
      let rightNorm = 0;
      for (let index = 0; index < embeddingSize; index += 1) {
        const a = flatEmbeddings[left * embeddingSize + index];
        const b = flatEmbeddings[right * embeddingSize + index];
        dot += a * b;
        leftNorm += a * a;
        rightNorm += b * b;
      }
      const similarity = dot / Math.max(Math.sqrt(leftNorm * rightNorm), 1e-12);
      matrix[left][right] = similarity;
      matrix[right][left] = similarity;
    }
  }
  return matrix;
}

export function completeLinkageClusters(
  similarityMatrix,
  squareIndices,
  threshold,
  crossBackgroundThreshold = null,
) {
  if (similarityMatrix.length !== squareIndices.length) {
    throw new Error("Similarity matrix and square indices must have the same size");
  }
  const groups = squareIndices.map((_, index) => [index]);
  while (groups.length > 1) {
    let bestLeft = -1;
    let bestRight = -1;
    let bestSimilarity = -Infinity;
    for (let left = 0; left < groups.length; left += 1) {
      for (let right = left + 1; right < groups.length; right += 1) {
        let completeSimilarity = Infinity;
        for (const a of groups[left]) {
          for (const b of groups[right]) {
            const firstSquare = squareIndices[a];
            const secondSquare = squareIndices[b];
            const firstColour = (Math.floor(firstSquare / 8) + firstSquare % 8) % 2;
            const secondColour = (Math.floor(secondSquare / 8) + secondSquare % 8) % 2;
            const boost = crossBackgroundThreshold !== null && firstColour !== secondColour
              ? threshold - crossBackgroundThreshold
              : 0;
            completeSimilarity = Math.min(
              completeSimilarity,
              similarityMatrix[a][b] + boost,
            );
          }
        }
        if (completeSimilarity > bestSimilarity) {
          bestSimilarity = completeSimilarity;
          bestLeft = left;
          bestRight = right;
        }
      }
    }
    if (bestSimilarity < threshold) break;
    groups[bestLeft] = [...groups[bestLeft], ...groups[bestRight]].sort((a, b) => a - b);
    groups.splice(bestRight, 1);
  }
  groups.sort((a, b) => squareIndices[a[0]] - squareIndices[b[0]]);
  return groups.map((members, groupId) => {
    let confidence = 1;
    for (let a = 0; a < members.length; a += 1) {
      for (let b = a + 1; b < members.length; b += 1) {
        confidence = Math.min(confidence, similarityMatrix[members[a]][members[b]]);
      }
    }
    return {
      groupId,
      squareIndices: members.map((index) => squareIndices[index]),
      confidence,
    };
  });
}

function logSoftmax(values) {
  const maximum = Math.max(...values);
  const normalizer = maximum + Math.log(
    values.reduce((sum, value) => sum + Math.exp(value - maximum), 0),
  );
  return values.map((value) => value - normalizer);
}

// Rectangular minimum-cost assignment for rows <= columns.
function hungarian(costs) {
  const rowCount = costs.length;
  if (rowCount === 0) return [];
  const columnCount = costs[0].length;
  if (rowCount > columnCount) throw new Error("Assignment requires at least as many slots as groups");
  const u = Array(rowCount + 1).fill(0);
  const v = Array(columnCount + 1).fill(0);
  const p = Array(columnCount + 1).fill(0);
  const way = Array(columnCount + 1).fill(0);
  for (let row = 1; row <= rowCount; row += 1) {
    p[0] = row;
    let column0 = 0;
    const minimum = Array(columnCount + 1).fill(Infinity);
    const used = Array(columnCount + 1).fill(false);
    do {
      used[column0] = true;
      const row0 = p[column0];
      let delta = Infinity;
      let column1 = 0;
      for (let column = 1; column <= columnCount; column += 1) {
        if (used[column]) continue;
        const current = costs[row0 - 1][column - 1] - u[row0] - v[column];
        if (current < minimum[column]) {
          minimum[column] = current;
          way[column] = column0;
        }
        if (minimum[column] < delta) {
          delta = minimum[column];
          column1 = column;
        }
      }
      for (let column = 0; column <= columnCount; column += 1) {
        if (used[column]) {
          u[p[column]] += delta;
          v[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      column0 = column1;
    } while (p[column0] !== 0);
    do {
      const column1 = way[column0];
      p[column0] = p[column1];
      column0 = column1;
    } while (column0 !== 0);
  }
  const result = Array(rowCount).fill(-1);
  for (let column = 1; column <= columnCount; column += 1) {
    if (p[column] > 0) result[p[column] - 1] = column - 1;
  }
  return result;
}

export function assignGroupLabels(
  squareLogits,
  groups,
  classCount,
  duplicatePenalty = 1.5,
  fixedLabels = new Map(),
) {
  if (groups.length === 0) return [];
  const groupLogits = groups.map((group) => {
    const mean = Array(classCount).fill(0);
    for (const squareIndex of group.squareIndices) {
      for (let classId = 0; classId < classCount; classId += 1) {
        mean[classId] += squareLogits[squareIndex][classId] / group.squareIndices.length;
      }
    }
    return mean;
  });
  const logProbabilities = groupLogits.map(logSoftmax);
  const slots = [];
  const kingIds = new Set([6, 12]);
  for (let classId = 0; classId < classCount; classId += 1) {
    const copies = kingIds.has(classId) ? 1 : groups.length;
    for (let duplicate = 0; duplicate < copies; duplicate += 1) {
      const penalty = classId === 0 ? 0 : duplicate * duplicatePenalty;
      slots.push({ classId, duplicate, penalty });
    }
  }
  const costs = groups.map((group, row) => slots.map((slot) => {
    const fixed = fixedLabels.get(group.groupId);
    if (fixed !== undefined && slot.classId !== fixed) return 1e9;
    return -(logProbabilities[row][slot.classId] - slot.penalty);
  }));
  const selected = hungarian(costs);
  return groups.map((group, row) => {
    const slot = slots[selected[row]];
    const probabilities = logProbabilities[row].map(Math.exp);
    return {
      groupId: group.groupId,
      classId: slot.classId,
      confidence: probabilities[slot.classId],
      probabilities,
      duplicate: slot.duplicate,
    };
  });
}
