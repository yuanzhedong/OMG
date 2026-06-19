from __future__ import annotations

from torch.utils.data import Dataset

from omg.data.datamodule import DistributedMaterializedShardSampler


class FakeMaterializedDataset(Dataset):
    def __init__(self, span_count: int = 8, span_size: int = 10) -> None:
        self.span_count = int(span_count)
        self.span_size = int(span_size)

    def __len__(self) -> int:
        return self.span_count * self.span_size

    def __getitem__(self, idx: int) -> int:
        return int(idx)

    def materialized_shard_spans(self, base_index: int = 0) -> list[tuple[int, int]]:
        return [
            (int(base_index) + span_idx * self.span_size, self.span_size)
            for span_idx in range(self.span_count)
        ]


def _span_id(index: int, span_size: int) -> int:
    return int(index) // int(span_size)


def _consecutive_runs(span_ids: list[int]) -> list[int]:
    """Return the distinct span id of each maximal run of equal consecutive ids."""
    runs: list[int] = []
    for span_id in span_ids:
        if not runs or runs[-1] != span_id:
            runs.append(span_id)
    return runs


def test_materialized_sampler_keeps_samples_shard_local(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    dataset = FakeMaterializedDataset(span_count=8, span_size=10)
    sampler = DistributedMaterializedShardSampler(dataset, seed=7)

    indices = list(iter(sampler))

    # Every sample is emitted exactly once.
    assert sorted(indices) == list(range(len(dataset)))
    assert len(sampler) == len(indices)

    # Consecutive samples stay within the same shard span: there is exactly one
    # contiguous run per span, and within a run indices are shard-local.
    span_ids = [_span_id(index, dataset.span_size) for index in indices]
    runs = _consecutive_runs(span_ids)
    assert len(runs) == dataset.span_count
    assert sorted(runs) == list(range(dataset.span_count))


def test_materialized_sampler_is_deterministic_per_seed(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    dataset = FakeMaterializedDataset(span_count=8, span_size=10)

    first = list(iter(DistributedMaterializedShardSampler(dataset, seed=7)))
    same = list(iter(DistributedMaterializedShardSampler(dataset, seed=7)))
    other = list(iter(DistributedMaterializedShardSampler(dataset, seed=8)))

    assert first == same
    assert first != other


def test_materialized_sampler_partitions_shards_across_ranks(monkeypatch) -> None:
    dataset = FakeMaterializedDataset(span_count=8, span_size=10)
    per_rank = []
    for rank in (0, 1):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "2")
        sampler = DistributedMaterializedShardSampler(dataset, seed=11)
        per_rank.append(list(iter(sampler)))

    # Ranks see disjoint samples whose union is the full dataset.
    assert set(per_rank[0]).isdisjoint(per_rank[1])
    assert sorted(per_rank[0] + per_rank[1]) == list(range(len(dataset)))

    # Each shard span is assigned wholly to a single rank (never split).
    for indices in per_rank:
        span_ids = [_span_id(index, dataset.span_size) for index in indices]
        runs = _consecutive_runs(span_ids)
        assert len(runs) == len(set(span_ids))
