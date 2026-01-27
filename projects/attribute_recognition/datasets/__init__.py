from .pa100k import PA100K

__datasets = {"pa100k": PA100K}


def init_dataset(name, **kwargs):
    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError(f'Invalid dataset name. Received "{name}", but expected to be one of {avai_datasets}')
    return __datasets[name](**kwargs)
