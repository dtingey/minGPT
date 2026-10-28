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

def eval_split(trainer, model, split, dataset, max_batches):
    n = len(dataset)
    results = []
    loader = DataLoader(dataset, batch_size=trainer.config.batch_size, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        if b >= max_batches:
            break
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        inp = x[:, :n]
        sol = y[:, -n:]

        cat = model.generate(inp, n, do_sample=False)
        sol_candidate = cat[:, n:]

        correct = (sol == sol_candidate).all(1).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            print("GPT claims that %s sorted is %s but gt is %s" % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist()))

    rt = torch.tensor(results, dtype=torch.float)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()

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
    train_config.max_iters = 101
    train_config.num_workers = 0
    train_config.batch_size = 1
    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    model.eval()

    with torch.no_grad():
        eval_split(trainer, model, 'train', train_dataset, max_batches=5)
        eval_split(trainer, model, 'test', test_dataset, max_batches=5)

    breakpoint()
    


    

if __name__ == "__main__":
    train()
