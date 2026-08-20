from __future__ import annotations

import torch

from chess_ocr.inference.piece_clusterer import PieceClusterer


def test_clusters_similar_embeddings_without_needing_cluster_count() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.05],
            [0.0, 1.0],
            [0.04, 0.99],
        ]
    )

    result = PieceClusterer(0.9).cluster(embeddings, [2, 7, 11, 19])

    assert [cluster.square_indices for cluster in result.clusters] == [(2, 7), (11, 19)]


def test_complete_linkage_prevents_similarity_chaining() -> None:
    # Angles 0, 20 and 40 degrees: adjacent pairs are above 0.9, endpoints are not.
    radians = torch.deg2rad(torch.tensor([0.0, 20.0, 40.0]))
    embeddings = torch.stack((torch.cos(radians), torch.sin(radians)), dim=1)

    result = PieceClusterer(0.9).cluster(embeddings)

    assert len(result.clusters) == 2
    assert sorted(len(cluster.square_indices) for cluster in result.clusters) == [1, 2]


def test_cross_background_threshold_only_relaxes_opposite_colour_pairs() -> None:
    # a8/b8 are opposite colours and have cosine similarity 0.95. a8/c8 are
    # the same colour with the same similarity, so only the former may merge.
    cosine = 0.95
    sine = (1.0 - cosine**2) ** 0.5
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [cosine, sine],
            [cosine, -sine],
        ]
    )

    result = PieceClusterer(0.98, 0.94).cluster(embeddings, [0, 1, 2])

    assert [cluster.square_indices for cluster in result.clusters] == [(0, 1), (2,)]


def test_empty_embedding_batch_returns_no_clusters() -> None:
    result = PieceClusterer(0.9).cluster(torch.empty((0, 64)), [])

    assert result.clusters == ()
    assert result.similarity_matrix.shape == (0, 0)
