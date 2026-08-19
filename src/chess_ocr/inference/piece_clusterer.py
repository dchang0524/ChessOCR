"""Conservative clustering of same-theme chess-piece embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PieceCluster:
    """One group of square indices believed to show the same piece sprite."""

    group_id: int
    square_indices: tuple[int, ...]
    confidence: float


@dataclass(frozen=True)
class ClusteringResult:
    """Clusters plus the full cosine-similarity matrix."""

    clusters: tuple[PieceCluster, ...]
    similarity_matrix: torch.Tensor


class PieceClusterer:
    """Complete-linkage agglomerative clustering with an automatic group count.

    Complete linkage merges two groups only when their least-similar cross-pair
    meets the threshold. This intentionally prefers false splits over false
    merges because group-wide user corrections make false merges more harmful.
    """

    def __init__(self, similarity_threshold: float) -> None:
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [-1, 1]")
        self.similarity_threshold = similarity_threshold

    def cluster(
        self, embeddings: torch.Tensor, square_indices: list[int] | tuple[int, ...] | None = None
    ) -> ClusteringResult:
        """Cluster selected embeddings and return original square indices."""
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2-D embeddings, got shape {tuple(embeddings.shape)}")
        count = embeddings.shape[0]
        indices = tuple(range(count)) if square_indices is None else tuple(square_indices)
        if len(indices) != count:
            raise ValueError("square_indices must align one-to-one with embeddings")
        if count == 0:
            return ClusteringResult((), torch.empty((0, 0)))

        normalised = F.normalize(embeddings.float(), p=2, dim=1)
        similarities = normalised @ normalised.T
        working: list[list[int]] = [[index] for index in range(count)]
        linkage = similarities.clone()
        linkage.fill_diagonal_(-float("inf"))

        while len(working) > 1:
            flat_index = int(linkage.argmax())
            width = linkage.shape[1]
            left, right = divmod(flat_index, width)
            best_similarity = float(linkage[left, right])
            if best_similarity < self.similarity_threshold:
                break
            if left > right:
                left, right = right, left
            working[left] = sorted(working[left] + working[right])
            working.pop(right)
            # Complete-link similarity between a merged cluster and another
            # cluster is the minimum of the previous two similarities.
            linkage[left] = torch.minimum(linkage[left], linkage[right])
            linkage[:, left] = linkage[left]
            linkage = torch.cat((linkage[:right], linkage[right + 1 :]), dim=0)
            linkage = torch.cat((linkage[:, :right], linkage[:, right + 1 :]), dim=1)
            linkage[left, left] = -float("inf")

        working.sort(key=lambda members: min(indices[index] for index in members))
        clusters: list[PieceCluster] = []
        for group_id, members in enumerate(working):
            if len(members) == 1:
                confidence = 1.0
            else:
                confidence = min(
                    float(similarities[members[a], members[b]])
                    for a in range(len(members))
                    for b in range(a + 1, len(members))
                )
            clusters.append(
                PieceCluster(
                    group_id=group_id,
                    square_indices=tuple(indices[index] for index in members),
                    confidence=confidence,
                )
            )
        return ClusteringResult(tuple(clusters), similarities.cpu())
