import numpy as np


def dataset_sampler(dataset, sampling_method='sqrt'):
    """Function to sample a single sub-dataset from a meta-dataset.

    Args:
        dataset (object): dataset class with .lens as property. This must be a dictionary with sub-datasets as keys and sizes as values.
        sampling_method (str, optional): sampling method to apply. 'uniform' gives every dataset equal weight. 'size' gives dataset weight according to dataset size. 'log'  weights according to log-dataset size. Similar to uniform in most cases. 'sqrt' weights according to sqrt-dataset size. Defaults to 'sqrt'.

    Returns:
        str: name of the sampled dataset.
    """
    sources, sizes = [], []
    for source, size in dataset.lens.items():
        sources.append(source)
        sizes.append(size)

    sizes = np.array(sizes)
    if sampling_method == 'uniform':
        sizes = np.ones_like(sizes)
    elif sampling_method == 'size':
        pass
    elif sampling_method == 'log':
        sizes = np.log(sizes)
    elif sampling_method == 'sqrt':
        sizes = np.sqrt(sizes)

    weights = sizes / np.sum(sizes)

    return np.random.choice(sources, p=weights)
