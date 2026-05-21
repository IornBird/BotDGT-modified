"""
To generate and test TwiBot20 stub
"""

import argparse
import json
import os
import pickle
import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torch_geometric.data import Data


def save_pt(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def compute_bidirectional_ratio(edge_index, num_nodes):
    src, dst = edge_index
    nbrs = [set() for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        nbrs[u].add(v)

    ratios = torch.zeros(num_nodes, dtype=torch.float32)
    for i in range(num_nodes):
        if len(nbrs[i]) == 0:
            continue
        reciprocal = sum(1 for j in nbrs[i] if i in nbrs[j])
        ratios[i] = float(reciprocal) / float(len(nbrs[i]))
    return ratios


def compute_clustering_coefficient(edge_index, num_nodes):
    src, dst = edge_index
    nbrs = [set() for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        nbrs[u].add(v)
        nbrs[v].add(u)

    coefficients = torch.zeros(num_nodes, dtype=torch.float32)
    for i in range(num_nodes):
        neighbors = list(nbrs[i])
        degree = len(neighbors)
        if degree < 2:
            continue
        links = 0
        for left in range(degree):
            for right in range(left + 1, degree):
                if neighbors[right] in nbrs[neighbors[left]]:
                    links += 1
        coefficients[i] = float(links) / float(degree * (degree - 1) / 2)
    return coefficients


def make_edges(num_nodes, snapshot_index):
    edges = []
    edge_types = []

    active_nodes = min(num_nodes, 8 + snapshot_index * 4)
    for i in range(active_nodes):
        j = (i + 1) % active_nodes
        edges.extend([(i, j), (j, i)])
        edge_types.extend([0, 0])

    for i in range(0, active_nodes, 3):
        j = (i + 3 + snapshot_index) % active_nodes
        if i != j:
            edges.append((i, j))
            edge_types.append(1)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    return edge_index, edge_type, active_nodes


def make_raw_users(num_nodes):
    users = []
    for i in range(num_nodes):
        created_year = 2009 + (i % 6)
        label = 1 if i % 5 == 0 else 0
        users.append({
            "id": f"u{i}",
            "ID": str(100000 + i),
            "created_at": datetime(created_year, (i % 12) + 1, 1).isoformat(),
            "profile": {
                "id_str": str(100000 + i),
                "screen_name": f"stub_user_{i}",
                "followers_count": 100 + i * 7,
                "friends_count": 80 + i * 3,
                "statuses_count": 50 + i * 11,
                "verified": i % 11 == 0,
                "default_profile": i % 2 == 0,
            },
            "tweet": [f"synthetic tweet {i}-{j}" for j in range(3)],
            "neighbor": {
                "following": [f"u{(i + 1) % num_nodes}", f"u{(i + 2) % num_nodes}"],
                "follower": [f"u{(i - 1) % num_nodes}", f"u{(i - 2) % num_nodes}"],
            },
            "domain": ["politics", "business", "entertainment", "sports"][i % 4],
            "label": str(label),
        })
    return users


def clean_stub_outputs(data_dir):
    for relative in [
        os.path.join("graph_data", "graphs"),
        os.path.join("final_data"),
    ]:
        path = os.path.join(data_dir, relative)
        if os.path.isdir(path):
            for name in os.listdir(path):
                if name.startswith("stub_") or name.endswith(".pt"):
                    target = os.path.join(path, name)
                    if os.path.isdir(target):
                        shutil.rmtree(target)
                    else:
                        os.remove(target)


def main():
    parser = argparse.ArgumentParser(description="Create a tiny TwiBot-20 compatible stub dataset.")
    parser.add_argument("--dataset-name", default="Twibot-20")
    parser.add_argument("--num-nodes", type=int, default=24)
    parser.add_argument("--num-snapshots", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--clean", action="store_true", help="Remove previous stub .pt outputs before writing.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    data_dir = os.path.join(root, "data", args.dataset_name)
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed_data")
    graph_dir = os.path.join(data_dir, "graph_data", "graphs")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    if args.clean:
        clean_stub_outputs(data_dir)
        os.makedirs(graph_dir, exist_ok=True)

    users = make_raw_users(args.num_nodes)
    with open(os.path.join(raw_dir, "node.json"), "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    with open(os.path.join(raw_dir, "user.json"), "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    with open(os.path.join(processed_dir, "uid2global_index.pkl"), "wb") as f:
        pickle.dump({user["id"]: i for i, user in enumerate(users)}, f)

    des_tensor = torch.randn(args.num_nodes, 768)
    tweets_tensor = torch.randn(args.num_nodes, 768)
    num_prop = torch.stack([
        torch.tensor([
            user["profile"]["followers_count"],
            user["profile"]["friends_count"],
            user["profile"]["statuses_count"],
            float(user["profile"]["verified"]),
            float(user["profile"]["default_profile"]),
        ], dtype=torch.float32)
        for user in users
    ])
    cat_prop = torch.stack([
        torch.tensor([
            float(user["profile"]["verified"]),
            float(user["profile"]["default_profile"]),
            float(i % 4),
        ], dtype=torch.float32)
        for i, user in enumerate(users)
    ])
    labels = torch.tensor([int(user["label"]) for user in users], dtype=torch.long)

    train_end = max(2, int(args.num_nodes * 0.6))
    val_end = max(train_end + 1, int(args.num_nodes * 0.8))
    train_idx = torch.arange(0, train_end, dtype=torch.long)
    val_idx = torch.arange(train_end, val_end, dtype=torch.long)
    test_idx = torch.arange(val_end, args.num_nodes, dtype=torch.long)

    save_pt(train_idx, os.path.join(processed_dir, "train_idx.pt"))
    save_pt(val_idx, os.path.join(processed_dir, "val_idx.pt"))
    save_pt(test_idx, os.path.join(processed_dir, "test_idx.pt"))
    save_pt(labels, os.path.join(processed_dir, "label.pt"))
    save_pt(des_tensor, os.path.join(processed_dir, "des_tensor.pt"))
    save_pt(tweets_tensor, os.path.join(processed_dir, "tweets_tensor.pt"))
    save_pt(num_prop, os.path.join(processed_dir, "num_properties_tensor.pt"))
    save_pt(cat_prop, os.path.join(processed_dir, "cat_properties_tensor.pt"))

    final_edge_index, final_edge_type, _ = make_edges(args.num_nodes, args.num_snapshots - 1)
    save_pt(final_edge_index, os.path.join(processed_dir, "edge_index.pt"))
    save_pt(final_edge_type, os.path.join(processed_dir, "edge_type.pt"))

    for snapshot_index in range(args.num_snapshots):
        edge_index, edge_type, active_nodes = make_edges(args.num_nodes, snapshot_index)
        exist_nodes = torch.zeros(args.num_nodes, dtype=torch.float32)
        exist_nodes[:active_nodes] = 1.0
        graph = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            exist_nodes=exist_nodes,
            clustering_coefficient=compute_clustering_coefficient(edge_index, args.num_nodes).reshape(-1, 1),
            bidirectional_links_ratio=compute_bidirectional_ratio(edge_index, args.num_nodes).reshape(-1, 1),
            n_id=torch.arange(args.num_nodes, dtype=torch.long),
            num_nodes=args.num_nodes,
        )
        save_pt(
            graph,
            os.path.join(graph_dir, f"stub_graph_in_snapshot_20{snapshot_index + 8:02d}-01-01.pt"),
        )

    print("TwiBot-20 stub dataset created")
    print(f" dataset: {data_dir}")
    print(f" nodes: {args.num_nodes}")
    print(f" graph snapshots: {args.num_snapshots}")
    print(f" train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")


if __name__ == "__main__":
    main()
