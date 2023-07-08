import pytorch_lightning as pl
from lightning.pytorch.utilities.data import DataLoader
from multiprocessing import cpu_count
import torch


class LitDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, test_set, tokenizer):
        super(LitDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = cpu_count()
        self.test_set = test_set
        self.tokenizer = tokenizer
        self.isAlpaca = "alpaca" in tokenizer.name_or_path

    def collate(self, batch):
        batch = [b.values() for b in batch]
        input_text, output_text, label_spaces = list(zip(*batch))
        assert len(input_text) == len(output_text) == len(label_spaces)

        batch = self.tokenizer(text=input_text, text_target=output_text, padding='longest', truncation=True,
                               return_tensors="pt", max_length=512)
        try:
            labels_cls = torch.ShortTensor([label_space.index(y) for label_space, y in zip(label_spaces, output_text)])
        except ValueError:
            print(label_spaces)
            print(output_text)
            raise

        label_spaces_ids = [self.tokenizer(label_space, padding=False, return_length=True) for label_space in label_spaces]
        sample_to = [label_space["length"] for label_space in label_spaces_ids]
        max_seq_len = max([max(leng) for leng in sample_to])
        label_spaces_ids = [self.tokenizer(label_space, padding="max_length", max_length=max_seq_len,
                                           return_tensors="pt")["input_ids"] for label_space in label_spaces]
        label_spaces_ids = torch.stack(label_spaces_ids, dim=0)

        sample_to = torch.ShortTensor([min(lengths) for lengths in sample_to])
        if self.isAlpaca:
            label_spaces_ids = label_spaces_ids[:, :, 1:] # remove the <s> token
            sample_to -= 1
            max_seq_len -= 1
            
        batch["label_cls"] = labels_cls
        batch["label_spaces_ids"] = label_spaces_ids
        batch["sample_to"] = sample_to
        return batch

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.collate)

