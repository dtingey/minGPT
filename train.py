from argparse import ArgumentParser
import math

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

def generate_samples(trainer, model, dataset, prompts, max_new_tokens=50):
    tokenizer = dataset.tokenizer.encoder
    context_window = model.block_size - max_new_tokens

    print("\n=== Sample generations ===")
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)[-context_window:]
        x = torch.tensor(prompt_ids, dtype=torch.long, device=trainer.device)[None, ...]
        generated = model.generate(x, max_new_tokens, do_sample=True, top_k=40)
        text = tokenizer.decode(generated[0].tolist())
        print(f"\nPrompt: {prompt!r}\n{text}")


def eval_loss(trainer, model, split, dataset, max_batches=5):
    loader = DataLoader(dataset, batch_size=trainer.config.batch_size, num_workers=0)
    losses = []

    for b, (x, y) in enumerate(loader):
        if b >= max_batches:
            break
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        _, loss = model(x, y)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    print(f"{split} loss: {mean_loss:.4f}; perplexity: {perplexity:.2f}")


def eval_generation(trainer, model, split, dataset, max_batches=5, max_new_tokens=32):
    loader = DataLoader(dataset, batch_size=trainer.config.batch_size, num_workers=0)
    prompt_length = dataset.get_block_size() // 2
    results = []

    for b, (x, _) in enumerate(loader):
        if b >= max_batches:
            break
        x = x.to(trainer.device)
        prompt = x[:, :prompt_length]
        target = x[:, prompt_length:prompt_length + max_new_tokens]
        generated = model.generate(prompt, target.size(1), do_sample=False)
        prediction = generated[:, prompt_length:]
        results.extend(prediction.eq(target).all(dim=1).cpu().tolist())

    accuracy = sum(results) / len(results)
    print(f"{split} generation: {sum(results)}/{len(results)} = {100 * accuracy:.2f}% correct")

def train():
    parser = ArgumentParser()
    parser.add_argument('--file_path', type=str, default='test_data.jsonl', help='Path to the JSONL dataset file')
    parser.add_argument('--test_size', type=int, default=20, help='Number of samples to use for testing')
    parser.add_argument('--block_size', type=int, default=128, help='Block size for the model')
    args = parser.parse_args()

    train_dataset = JSONLDataset(file_path='test_data.jsonl', split='train', test_size=args.test_size, block_size=args.block_size)
    test_dataset = JSONLDataset(file_path='test_data.jsonl', split='test', test_size=args.test_size, block_size=args.block_size)

    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 10_001
    train_config.num_workers = 0
    train_config.batch_size = 1
    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    model.eval()

    with torch.no_grad():
        eval_loss(trainer, model, 'train', train_dataset)
        eval_loss(trainer, model, 'test', test_dataset)
        eval_generation(trainer, model, 'train', train_dataset)
        eval_generation(trainer, model, 'test', test_dataset)
        generate_samples(
            trainer,
            model,
            train_dataset,
            prompts=[
                "The",
                "Once upon a time",
                "In the future",
                "The most important thing",
            ],
        )

if __name__ == "__main__":
    train()
