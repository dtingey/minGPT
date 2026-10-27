##################################################
# DATASET                                        #
##################################################

import json

import torch
from torch.utils.data import Dataset

from mingpt.bpe import BPETokenizer


class JSONLDataset(Dataset):
    def __init__(self, file_path, split='train', test_size=1000, block_size=128):
        self.file_path = file_path
        self.block_size = block_size
        self.tokenizer = BPETokenizer()
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        print(f'Length of lines: {len(lines)}')
        prelength = len(lines)
        data = []
        for i, line in enumerate(lines):
            try:
                # Processes the line and extracts the 'text' field
                d = json.loads(line)['text']
                # Skips short text entries
                if len(d) < 10:
                    continue
                data.append(d)
            except json.JSONDecodeError:
                # Silently skip lines that fail to parse
                pass 

        print(f'Length of data: {len(data) / prelength * 100:.2f}%')
        
        # Implements the train/test split, then tokenize it as one continuous stream.
        if split == 'train':
            data = data[:-test_size]
        else:
            data = data[-test_size:]

        eos_id = self.tokenizer.encoder.encoder['<|endoftext|>']
        self.data = []
        for text in data:
            self.data.extend(self.tokenizer.encoder.encode(text))
            self.data.append(eos_id)

    def __len__(self):
        # Each item needs block_size + 1 tokens to form inputs and targets.
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def get_vocab_size(self):
        return len(self.tokenizer.encoder.encoder)  # 50,257 for GPT-2

    def get_block_size(self):
        return self.block_size


if __name__ == '__main__':
    file_path = 'test_data.jsonl'
# file_path = 'pile_data_10_first_50000.jsonl'
# file_path = '100.jsonl'

# Initialize the dataset with a test size of 1000 lines
# print(f'Making dataset... {get_elapsed_time() / 60} minutes')
    train_dataset = JSONLDataset(file_path, split='train', test_size=10)
    test_dataset = JSONLDataset(file_path, split='test', test_size=10)
    print(len(train_dataset))
    print(len(test_dataset))
