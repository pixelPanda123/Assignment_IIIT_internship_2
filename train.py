import os
import argparse
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from data import LMTextDataset
from model import GPTLanguageModel
import config
from utils import set_seed, ensure_dir, save_plot, save_json, perplexity

def evaluate(model, dataset, device, batch_size, block_size, batches=200):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = dataset.get_batch('val', batch_size, block_size, device)
            logits = model(x)
            loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
    return total / batches

def evaluate_test(model, dataset, device, batch_size, block_size, batches=400):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = dataset.get_batch('test', batch_size, block_size, device)
            logits = model(x)
            loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
    return total / batches

def train_loop(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ds = LMTextDataset(args.dataset_path, lowercase=args.lowercase)
    cfg = config.get(args.exp)
    d_model = cfg['d_model']
    n_layers = cfg['n_layers']
    n_heads = cfg['n_heads']
    dropout = cfg['dropout']
    lr = cfg['lr']
    weight_decay = cfg['weight_decay']
    epochs = args.epochs or cfg['epochs']
    batch_size = args.batch_size or cfg['batch_size']
    block_size = args.block_size or cfg['block_size']
    warmup_steps = cfg['warmup_steps']
    patience = cfg['patience']
    model = GPTLanguageModel(256, d_model, n_layers, n_heads, block_size, dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    scaler = amp.GradScaler(enabled=args.amp)
    ce = nn.CrossEntropyLoss()
    steps_per_epoch = max(1, (ds.train_tokens.size(0) - block_size) // (batch_size * block_size))
    total_steps = steps_per_epoch * max(1, epochs)
    best_val = None
    best_path = os.path.join('checkpoints', args.exp, 'best.pt')
    ensure_dir(os.path.dirname(best_path))
    train_losses = []
    val_losses = []
    warmup = warmup_steps
    def lr_schedule(step):
        if warmup > 0 and step < warmup:
            return lr * (step + 1) / warmup
        t = step - warmup
        T = max(1, total_steps - warmup)
        return 1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * t / T))
    no_improve = 0
    step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for _ in range(steps_per_epoch):
            x, y = ds.get_batch('train', batch_size, block_size, device)
            for g in opt.param_groups:
                g['lr'] = lr_schedule(step)
            opt.zero_grad(set_to_none=True)
            if args.amp:
                with amp.autocast():
                    logits = model(x)
                    loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
                if args.clip_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = ce(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                if args.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                opt.step()
            running += loss.item()
            step += 1
        train_loss = running / steps_per_epoch
        val_loss = evaluate(model, ds, device, batch_size, block_size, batches=args.val_batches)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'cfg': cfg}, best_path)
            no_improve = 0
        else:
            no_improve += 1
        if patience > 0 and no_improve >= patience:
            break
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])
    test_loss = evaluate_test(model, ds, device, batch_size, block_size, batches=args.test_batches)
    metrics = {
        'train_loss': train_losses[-1] if train_losses else None,
        'val_loss': val_losses[-1] if val_losses else None,
        'test_loss': test_loss,
        'val_perplexity': perplexity(val_losses[-1]) if val_losses else None,
        'test_perplexity': perplexity(test_loss)
    }
    save_plot(train_losses, val_losses, os.path.join('plots', f'{args.exp}.png'), args.exp)
    save_json(metrics, os.path.join('runs', args.exp, 'metrics.json'))
    if args.generate:
        start = torch.tensor([[ord(' ')]], dtype=torch.long, device=device)
        gen = model.generate(start, args.gen_tokens, temperature=args.temperature, top_k=args.top_k)
        text = ds.tokenizer.decode(gen[0].tolist())
        ensure_dir('runs')
        with open(os.path.join('runs', args.exp, 'sample.txt'), 'w') as f:
            f.write(text)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', type=str, default='bestfit')
    p.add_argument('--dataset_path', type=str, default=os.path.join('dataset', 'Pride_and_Prejudice-Jane_Austen.txt'))
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--block_size', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--clip_norm', type=float, default=1.0)
    p.add_argument('--val_batches', type=int, default=200)
    p.add_argument('--test_batches', type=int, default=400)
    p.add_argument('--lowercase', action='store_true')
    p.add_argument('--generate', action='store_true')
    p.add_argument('--gen_tokens', type=int, default=500)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=None)
    args = p.parse_args()
    train_loop(args)

if __name__ == '__main__':
    main()