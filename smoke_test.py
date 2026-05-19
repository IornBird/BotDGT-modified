from types import SimpleNamespace
import importlib
import sys

def check_import(name):
    try:
        mod = importlib.import_module(name)
        print(f"OK: imported {name}")
        return True, mod
    except Exception as e:
        print(f"ERR: import {name} failed: {e}")
        return False, None

# 1) Basic runtime checks
ok_torch, torch_mod = check_import('torch')
if ok_torch:
    try:
        print('torch.__version__ =', torch_mod.__version__)
    except Exception:
        pass

ok_tg, tg_mod = check_import('torch_geometric')
ok_tg_nn, tg_nn_mod = check_import('torch_geometric.nn')

# 2) Try importing core project module
ok_model, model_mod = check_import('models.model')

# 3) If import succeeded, attempt to instantiate the model with safe CPU args
if ok_model:
    try:
        args = SimpleNamespace()
        args.hidden_dim = 16
        args.structural_head_config = 2
        args.structural_drop = 0.0
        args.temporal_head_config = 2
        args.temporal_drop = 0.0
        args.window_size = 3
        args.temporal_module_type = 'attention'
        args.device = 'cpu'
        model = model_mod.BotDyGNN(args)
        print('OK: BotDyGNN instantiated')
    except Exception as e:
        print('ERR: BotDyGNN instantiation failed:', e)

print('\nSmoke test completed')
