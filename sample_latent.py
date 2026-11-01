"""Interactive text generation with the experimental LatentGPT sampler."""

import argparse

import torch

from mingpt.bpe import get_encoder
from mingpt.latent_model import LatentGPT


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gpt2",
        choices=("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"),
        help="Pretrained GPT-2 checkpoint to initialize LatentGPT (default: gpt2).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=80,
        help="Number of latent steps to decode for each prompt (default: 80).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature; lower values are less random (default: 0.8).",
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Restrict each decoded step to the top K tokens (default: 40; use 0 to disable).",
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="Choose the most likely token instead of sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens must be non-negative")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be greater than zero")
    if args.top_k < 0:
        raise SystemExit("--top-k must be non-negative")

    device = get_device()
    print(f"Loading {args.model} on {device}...")
    model = LatentGPT.from_pretrained(args.model).to(device).eval()
    encoder = get_encoder()
    top_k = args.top_k or None
    print("Ready. Type a prompt, or 'exit' to quit.")

    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        if not prompt.strip():
            continue

        token_ids = encoder.encode(prompt)
        if len(token_ids) > model.block_size:
            print(f"Prompt is {len(token_ids)} tokens; maximum context is {model.block_size}.")
            continue

        idx = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        print(encoder.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
