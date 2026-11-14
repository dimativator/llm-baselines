"""Run the repository trainer with all model serialization prohibited."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import torch
from main import get_args, main


def reject_checkpoint(*args, **kwargs):
    raise RuntimeError('Model and optimizer checkpoint writes are prohibited for this sweep')


if __name__ == '__main__':
    torch.save = reject_checkpoint
    args, parser = get_args()
    assert args.minimal_metrics and not args.wandb
    assert args.latest_ckpt_interval == args.permanent_ckpt_interval == 0
    assert args.hf_checkpoint_repo is None
    if args.device == 'cpu':
        torch.set_num_threads(2)
    else:
        assert torch.cuda.device_count() == 1, 'Exactly one allocated GPU required'
        print('GPU=' + str(torch.cuda.get_device_properties(0)), flush=True)
    main(args, parser)
