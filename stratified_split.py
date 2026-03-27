import numpy as np


def stratified_sample_items(items, labels, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    total = len(labels)

    selected = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        n_c = round(len(idx_c) / total * n_samples)
        n_c = min(n_c, len(idx_c))
        if n_c > 0:
            chosen = rng.choice(idx_c, size=n_c, replace=False)
            selected.append(chosen)

    if selected:
        idx_all = np.concatenate(selected)
    else:
        idx_all = np.array([], dtype=np.int64)

    if len(idx_all) < n_samples:
        remaining = np.setdiff1d(np.arange(total), idx_all)
        extra = rng.choice(remaining, size=min(n_samples - len(idx_all), len(remaining)), replace=False)
        idx_all = np.concatenate([idx_all, extra])
    elif len(idx_all) > n_samples:
        idx_all = rng.choice(idx_all, size=n_samples, replace=False)

    rng.shuffle(idx_all)
    sampled_items = [items[i] for i in idx_all]
    sampled_labels = [labels[i] for i in idx_all]
    return sampled_items, sampled_labels


def stratified_split_items(items, labels, n_workers, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    worker_indices = [[] for _ in range(n_workers)]
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        for i, idx in enumerate(idx_c):
            worker_indices[i % n_workers].append(idx)

    batches = []
    for wid in range(n_workers):
        w_idx = np.array(worker_indices[wid], dtype=np.int64)
        rng.shuffle(w_idx)
        batches.append([
            items[i] for i in w_idx
        ])
    return batches


def print_distribution_from_items(batches, max_classes_to_show=20):
    print("\n  Distribución de clases por worker")
    for wid, batch in enumerate(batches):
        labels = [label for _, label in batch]
        unique, counts = np.unique(labels, return_counts=True)
        stats = dict(zip(unique.tolist(), counts.tolist()))
        preview = [f"{cls}:{stats.get(cls, 0)}" for cls in sorted(stats.keys())[:max_classes_to_show]]
        suffix = "  ..." if len(stats) > max_classes_to_show else ""
        print(f"  W{wid:<3} total={len(batch):>6} | {'  '.join(preview)}{suffix}")
