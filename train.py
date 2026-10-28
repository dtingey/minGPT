import torch
from torch.utils.data import DataLoader

from dataset import JSONLDataset
from mingpt.utils import set_seed
from mingpt.model import GPT
from mingpt.trainer import Trainer

set_seed(3407)

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

def train():
    train_dataset = JSONLDataset(file_path='test_data.jsonl', split='train', test_size=10, block_size=128)
    test_dataset = JSONLDataset(file_path='test_data.jsonl', split='test', test_size=10, block_size=128)

    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 2000
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()


    

if __name__ == "__main__":
    train()
