import argparse

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--scenario-name", type=str, default="mec-test1", help="")
    parser.add_argument("--lr-actor", type=float, default=0.0001, help="")
    parser.add_argument("--lr-critic", type=float, default=0.0002, help="")
    parser.add_argument("--epsilon", type=float, default=0.1, help="")
    parser.add_argument("--gamma", type=float, default=0.99, help="")
    parser.add_argument("--tau", type=float, default=0.01, help="")
    parser.add_argument("--buffer-size", type=int, default=int(10000),help="")
    parser.add_argument("--batch-size", type=int, default=64, help="")
    parser.add_argument("--save-dir", type=str, default="./model", help="")
    parser.add_argument("--model-dir", type=str, default="", help="")
    args = parser.parse_args()

    return args
