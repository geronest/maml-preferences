import random
from torch.utils.data import IterableDataset, DataLoader

class MAMLDataset(IterableDataset):
    def __init__(self, dataset, batch_size, num_tasks_per_batch, steps_per_epoch=1000, sample_all=False):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_tasks_per_batch = num_tasks_per_batch
        self.steps_per_epoch = steps_per_epoch
        self.keys = list(dataset.keys())
        self.sample_all = sample_all
        self.prepare_dataloaders()

    def prepare_dataloaders(self):
        self.dataloaders = {
            lang: DataLoader(self.dataset[lang], batch_size=self.batch_size, shuffle=True) for lang in self.keys
        }

    def __iter__(self):
        # Create persistent iterators for each language for this epoch
        self.data_iterators = {
            lang: iter(loader) for lang, loader in self.dataloaders.items()
        }
        return self

    def __next__(self):
        # Randomly sample languages for this batch
        if self.sample_all:
            selected_langs = self.keys
        else:
            selected_langs = random.sample(self.keys, self.num_tasks_per_batch)
        
        # Create a batch with one example from each selected language
        return self.get_from_tasks(selected_langs)

    def get_from_tasks(self, tasks):
        batch = {}
        for lang in tasks:
            try:
                examples = next(self.data_iterators[lang])
            except StopIteration:
                # Re-initialize the iterator if it's exhausted
                self.data_iterators[lang] = iter(self.dataloaders[lang])
                examples = next(self.data_iterators[lang])

            batch[lang] = {
                key: examples[key] for key in examples.keys()
            }
        
        return batch