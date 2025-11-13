def get(exp):
    if exp == 'underfit':
        return {
            'd_model': 128,
            'n_layers': 2,
            'n_heads': 4,
            'dropout': 0.3,
            'lr': 3e-4,
            'weight_decay': 0.1,
            'epochs': 10,
            'batch_size': 64,
            'block_size': 256,
            'warmup_steps': 0,
            'patience': 0
        }
    if exp == 'overfit':
        return {
            'd_model': 512,
            'n_layers': 8,
            'n_heads': 8,
            'dropout': 0.0,
            'lr': 3e-4,
            'weight_decay': 0.0,
            'epochs': 40,
            'batch_size': 64,
            'block_size': 256,
            'warmup_steps': 0,
            'patience': 0
        }
    return {
        'd_model': 384,
        'n_layers': 6,
        'n_heads': 6,
        'dropout': 0.1,
        'lr': 3e-4,
        'weight_decay': 0.1,
        'epochs': 30,
        'batch_size': 64,
        'block_size': 256,
        'warmup_steps': 2000,
        'patience': 6
    }