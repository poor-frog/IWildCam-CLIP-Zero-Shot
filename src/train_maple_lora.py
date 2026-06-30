from src.config import parse_arguments
from src.train_maple_full import main


def configure_maple_lora_args(args):
    if args.maple_lora_rank <= 0:
        args.maple_lora_rank = 8
    if args.maple_lora_alpha is None:
        args.maple_lora_alpha = args.maple_lora_rank * 2
    if args.maple_lora_layers is None:
        args.maple_lora_layers = "last6"
    args.training_method = "maple_lora"
    if args.wandb_run_name is None:
        args.wandb_run_name = f"maple-lora-vit-b32-r{args.maple_lora_rank}-{args.maple_lora_layers}"
    if args.save is None:
        args.save = f"./checkpoints/maple_lora_r{args.maple_lora_rank}_{args.maple_lora_layers}.pt"
    return args


if __name__ == "__main__":
    import copy as _copy
    from src.train_coop import _inject_seed_suffix, print_aggregated_summary

    args = configure_maple_lora_args(parse_arguments())
    if args.runs <= 1:
        main(args)
    else:
        all_summaries = []
        base_seed = args.seed
        for run_idx in range(args.runs):
            print(f"\n{'='*60}")
            print(f"Run {run_idx + 1}/{args.runs} (seed={base_seed + run_idx})")
            print(f"{'='*60}")
            run_args = _copy.deepcopy(args)
            run_args.seed = base_seed + run_idx
            _inject_seed_suffix(run_args, run_idx)
            summary = main(run_args)
            all_summaries.append(summary)
        print_aggregated_summary(all_summaries)
