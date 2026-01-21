import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments for influence maximization and active learning experiments.")

    # Dataset parameters
    parser.add_argument("--dataset-name", type=str, default="imdb", help="Name of the dataset to use.")

    # IC model parameters
    parser.add_argument("--beta", type=float, default=0.55, help="Transmission probability (0.55 for academic networks, 0.85 for social networks).")
    parser.add_argument("--num-steps", type=int, default=3, help="Number of diffusion epochs (5-20).")
    parser.add_argument("--n-sim", type=int, default=3, help="Number of independent Monte-Carlo simulations per seed (200+ for research, 20–50 for quick experiments).")
    parser.add_argument("--reduction-factor", type=int, default=1, help="Reduction factor for graph size (e.g., 4-8).")
    parser.add_argument("--num-hops", type=int, default=2, help="Number of hops for computation.")

    # Model parameters
    parser.add_argument("--hidden-channels", type=int, default=3, help="Number of hidden channels.")
    parser.add_argument("--out-channels", type=int, default=3, help="Number of output channels.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers in the model.")

    # Data parameters
    parser.add_argument("--influence-levels", type=int, default=3, help="Number of influence levels (e.g., low, medium, high).")

    # Training parameters
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay for optimization.")
    parser.add_argument("--training-epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--percentage-training-set", type=float, default=0.8, help="Percentage of data to use for training.")
    parser.add_argument("--percentage-labeled-set", type=float, default=0.2, help="Percentage of data that is labeled.")
    parser.add_argument("--al-cycles", type=int, default=10, help="Number of Active Learning cycles.")
    parser.add_argument("--al-technique", type=str, default="MarginALTechnique", help="Active Learning technique class name.")
    parser.add_argument("--k", type=int, default=500, help="Number of nodes to sample per AL cycle.")

    return parser.parse_args()
