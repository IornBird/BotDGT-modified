# BotDGT Project Analysis Report

## 1. Project Overview

**Project Name:** BotDGT (Bot Detection with Dynamic Graph Transformers)

**Purpose:** A framework for detecting bot accounts in social networks using dynamic graph neural networks that leverage temporal information. The system detects whether Twitter accounts are bots or real users by analyzing their behavioral patterns, network structure, and temporal evolution.

**Reference Paper:** "BotDGT: Dynamicity-aware Social Network Bot Detection with Dynamic Graph Transformers" (IJCAI 2024)

**Supported Datasets:** 
- Twibot-20
- Twibot-22

---

## 2. System Architecture & Data Flow

### 2.1 Overall Workflow Pipeline

```
Raw Data (JSON)
    ↓
Data Preprocessing (generates graph snapshots)
    ↓
Feature Extraction (descriptions, tweets, numerical/categorical properties)
    ↓
Dataset Loading (train/val/test splits, batching)
    ↓
Model Forward Pass:
    ├─ NodeFeatureEmbeddingLayer (embeds multi-modal features)
    ├─ GraphStructuralLayer (processes graph structure for each snapshot)
    └─ GraphTemporalLayer (learns temporal dependencies across snapshots)
    ↓
Loss Computation (with temporal weighting)
    ↓
Optimization & Training
    ↓
Evaluation (Accuracy, F1, Precision, Recall)
```

### 2.2 Execution Flow (main.py → Trainer Class)

1. **Initialization Phase:**
   - Load configuration arguments from command line
   - Initialize Dataset object with specified dataset name, interval, batch size, etc.
   - Load features: description tensors, tweet tensors, numerical properties, categorical properties
   - Load graph snapshots (temporal sequence of graphs at different time intervals)
   - Split data into train/val/test sets
   - Create batched data loader for efficient processing

2. **Model Construction Phase:**
   - Build BotDyGNN model with 5 key components
   - Configure separate learning rates for structural and temporal components
   - Setup AdamW optimizer with weight decay
   - Setup cosine annealing learning rate scheduler

3. **Training Loop (per epoch):**
   - For each batch in training data:
     - **forward_one_batch()**: Process one batch through model
       - Extract multi-modal features for all nodes in batch
       - Pass through NodeFeatureEmbeddingLayer
       - Process through GraphStructuralLayer for each snapshot
       - Add position embeddings (clustering coefficient, bidirectional links)
       - Process through GraphTemporalLayer
     - Compute loss with temporal weighting
     - Backward pass
     - Optimizer step
   - Compute metrics on entire epoch

4. **Validation & Testing:**
   - Similar to training but without gradient computation
   - Track best validation metrics
   - Save checkpoint states

---

## 3. Key Model Components

### 3.1 BotDyGNN (Main Model Architecture)

**File:** `models/model.py`

**Architecture:**
```
Input (Multi-modal node features, Graph structure, Temporal snapshots)
    ↓
For each timestamp t in window_size:
    ├─ NodeFeatureEmbeddingLayer
    │   └─ Output: Node embeddings [B, hidden_dim]
    ├─ GraphStructuralLayer
    │   └─ Output: Graph-aware node embeddings [B, hidden_dim]
    └─ Store outputs: all_snapshots_structural_output [B, T, hidden_dim]
    ↓
Position Encodings:
    ├─ PositionEncodingClusteringCoefficient (network topology)
    └─ PositionEncodingBidirectionalLinks (edge directionality)
    ↓
GraphTemporalLayer:
    └─ Learns temporal dependencies across snapshots
    └─ Output: temporal_output [B, T, 2] (2 classes: bot/human)
    ↓
Classification Output
```

**Key Parameters:**
- `hidden_dim`: Embedding dimension (default: 128)
- `structural_head_config`: Attention heads in structural layer (default: 4)
- `temporal_head_config`: Attention heads in temporal layer (default: 4)
- `window_size`: Number of temporal snapshots to process

### 3.2 NodeFeatureEmbeddingLayer

**File:** `models/NodeFeatureEmbeddingLayer.py`

**Purpose:** Fuses multi-modal node features into a unified embedding representation

**Input Features:**
1. **Numerical Properties** (5 dimensions):
   - Account-level statistics (e.g., followers, following count, tweet count)
   
2. **Categorical Properties** (3 dimensions):
   - Categorical account attributes (e.g., account verification status, location info)
   
3. **Description Embeddings** (768 dimensions):
   - Pre-computed BERT embeddings from user profile descriptions
   
4. **Tweet Embeddings** (768 dimensions):
   - Pre-computed embeddings of user's recent tweets

**Processing:**
```
Each feature type:
    ↓
Linear transformation to hidden_dim/4
    ↓
PReLU activation
    ↓
    [num_features: hidden_dim/4, cat_features: hidden_dim/4, 
     des_features: hidden_dim/4, tweet_features: hidden_dim/4]
    ↓
Concatenate: [hidden_dim]
    ↓
Final linear layer + PReLU
    ↓
Output: Node embedding [B, hidden_dim]
```

### 3.3 GraphStructuralLayer

**File:** `models/GraphStructuralLayer.py`

**Purpose:** Capture graph-structural information for a single snapshot using attention-based convolution

**Architecture:**
```
Input: x [B, hidden_dim], edge_index [2, E]
    ↓
TransformerConv Layer 1 (multi-head attention convolution):
    - n_heads attention heads
    - Output dim: hidden_dim
    ↓
PReLU activation
    ↓
TransformerConv Layer 2 (another round of message passing)
    - Output dim: hidden_dim
    ↓
Residual connection: x + output_layer2
    ↓
PReLU activation
    ↓
Output: Graph-aware embeddings [B, hidden_dim]
```

**Key Mechanism:**
- Uses Transformer-based graph convolution from PyG (torch_geometric)
- Multi-head attention allows parallel learning of different structural patterns
- Residual connections improve gradient flow

### 3.4 GraphTemporalLayer

**File:** `models/GraphTemporalLayer.py`

**Purpose:** Model temporal dynamics across multiple graph snapshots

**Three Operational Modes:**

1. **GRU Mode** (`temporal_module_type='gru'`):
   - GRU processes sequence of structural embeddings
   - Output: `gru_out + structural_input` (residual)
   - Final classification head

2. **LSTM Mode** (`temporal_module_type='lstm'`):
   - LSTM processes sequence of structural embeddings
   - Output: `lstm_out + structural_input` (residual)
   - Final classification head

3. **Attention Mode** (`temporal_module_type='attention'`):
   - Multi-head self-attention with temporal masking
   - Incorporates position embeddings:
     - Temporal position encoding
     - Clustering coefficient position encoding
     - Bidirectional links ratio position encoding
   - **Temporal Masking:** Causal attention (tokens can only attend to earlier or current timesteps)
   - Architecture:
     ```
     Input: [B, T, hidden_dim]
         ↓
     Layer norm
         ↓
     Position embeddings fusion:
         + Temporal embeddings
         + Clustering coefficient embeddings
         + Bidirectional links ratio embeddings
         ↓
     Multi-head attention (Q,K,V projections)
         ↓
     Causal mask application
         ↓
     Softmax + attention dropout
         ↓
     Attention output fusion
         ↓
     Feed-forward network
         ↓
     Output: [B, T, 2] (bot/human classification logits)
     ```

**Position Encoding Layers:**
- `PositionEncodingClusteringCoefficient`: Encodes local network connectivity
- `PositionEncodingBidirectionalLinks`: Encodes bidirectional vs unidirectional links

---

## 4. Data Format & Structure

### 4.1 Input Data Formats

**Dataset Structure:**
```
data/
├── Twibot-20/
│   ├── raw/
│   │   ├── node.json          # Original Twitter data
│   │   └── user.json          # Extracted user data
│   ├── processed_data/        # Feature tensors
│   │   ├── des_tensor.pt                          # [N, 768] - Description embeddings
│   │   ├── tweets_tensor.pt                       # [N, 768] - Tweet embeddings
│   │   ├── num_properties_tensor.pt               # [N, 5]   - Numerical features
│   │   ├── cat_properties_tensor.pt               # [N, 3]   - Categorical features
│   │   ├── label.pt                               # [N] - Binary labels (0=human, 1=bot, 2=suspended, 3=unknown)
│   │   ├── train_idx.pt, val_idx.pt, test_idx.pt # Indices for splits
│   │   ├── edge_index.pt                          # [2, E] - Edge list (COO format)
│   │   ├── edge_type.pt                           # [E] - Edge types
│   │   └── uid2global_index.pkl                   # Mapping dict
│   ├── graph_data/
│   │   ├── graphs/                               # Per-snapshot processed graphs
│   │   │   ├── global_index_in_snapshot_YYYY-MM-DD.pt
│   │   │   ├── edge_index_in_snapshot_YYYY-MM-DD.pt
│   │   │   └── ...
│   │   ├── final_data/
│   │   │   └── year/batch-size-64/seed-1234/    # Cached batches
│   │   │       ├── train/all_right.pt, all_n_id.pt, ...
│   │   │       ├── val/
│   │   │       └── test/
│   └── output/                                    # Training checkpoints
```

### 4.2 Tensor Data Formats

**Node Features:**
```python
des_tensor:            torch.Tensor [N, 768]   # Description embeddings (BERT)
tweets_tensor:         torch.Tensor [N, 768]   # Tweet embeddings (BERT)
num_prop:              torch.Tensor [N, 5]     # Numerical properties
  - Dimensions: [followers_count, following_count, tweet_count, ...]
category_prop:         torch.Tensor [N, 3]     # Categorical properties
labels:                torch.Tensor [N]        # Labels: 0 (human), 1 (bot), 2 (suspended), 3 (unknown)
```

**Graph Structure (per snapshot):**
```python
edge_index:            torch.Tensor [2, E]     # Edge indices in COO format
                                                # edge_index[0] = source nodes
                                                # edge_index[1] = target nodes
edge_type:             torch.Tensor [E]        # Relationship types
                                                # Types: follow, mention, reply, retweet, etc.
clustering_coefficient: torch.Tensor [N]       # Local clustering coefficient per node
bidirectional_links_ratio: torch.Tensor [N]    # Ratio of bidirectional edges for each node
exist_nodes:           torch.Tensor [B, T]     # Binary mask: whether node exists in snapshot
                                                # Shape: [batch_size, window_size]
```

**Batch Data Format (from Dataset class):**
```python
# For each batch:
batch_size:                          int        # Actual batch size (≤ 64)
batch_n_id:                     List[Tensor]    # List of T tensors, each [B] - node indices per snapshot
batch_edge_index:               List[Tensor]    # List of T tensors, each [2, E_t] - edges per snapshot
batch_edge_type:                List[Tensor]    # List of T tensors, each [E_t] - edge types per snapshot
batch_exist_nodes:              List[Tensor]    # List of T tensors, each [B] - existence mask per snapshot
batch_clustering_coefficient:   List[Tensor]    # List of T tensors, each [B] - coefficients per snapshot
batch_bidirectional_links_ratio: List[Tensor]   # List of T tensors, each [B] - ratios per snapshot
```

### 4.3 Label Definitions

```
0: Human/Benign account
1: Bot account  
2: Suspended (account no longer active)
3: Unknown (unlabeled/uncertain)
```

For Twibot-20: During training, unknown accounts (label=3) are added as a third class. The model trains on a 3-way classification problem with 229,580 unknown accounts padded with label 3.

### 4.4 Time Intervals

Supported temporal interval options:
- `'year'`: Snapshots at 12-month intervals
- `'month'`: Monthly snapshots
- `'three_months'`, `'six_months'`, `'9_months'`: Quarterly/semi-annual intervals
- `'15_months'`, `'18_months'`, `'21_months'`, `'24_months'`: Custom intervals

---

## 5. Training Process & Loss Function

### 5.1 Loss Computation

**File:** `utils/loss.py`

**Temporal Weighted Loss:**
```
For each snapshot t in [0, T-1]:
    loss_t = CrossEntropyLoss(output[t], label[t])  # Only on existing nodes
    
Total Loss = Σ(loss_t * coefficient^t) for all snapshots
where coefficient = 1.1 (default)
```

**Key Features:**
- **Temporal Weighting:** Later snapshots have higher loss weights (exponential weighting with coefficient 1.1)
- **Mask Filtering:** Only computes loss for nodes that exist in each snapshot (`exist_nodes == 1`)
- **Averaging:** Loss is normalized by window size and number of snapshots

**Intuition:** Recent behavior is more important than historical behavior for bot detection, hence later snapshots get higher weights.

### 5.2 Optimization Strategy

**Optimizer:** AdamW with separate learning rates
```python
params = [
    {"params": structural_components, "lr": 1e-4},    # Structural learning rate
    {"params": temporal_components, "lr": 1e-5},      # Temporal learning rate (lower)
]
optimizer = AdamW(params, weight_decay=0.01)
scheduler = CosineAnnealingLR(T_max=20, eta_min=0)
```

**Rationale:** Temporal components learn at a lower rate to ensure stable temporal pattern learning.

---

## 6. Evaluation Metrics

**File:** `utils/metrics.py`

**Metrics Computed:**
- **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)

**Evaluation Strategy:**
- Computed only on the **last snapshot** of each batch (most recent temporal state)
- Only nodes with `exist_nodes == 1` are included in metrics
- Metrics are accumulated across all batches to compute epoch-level statistics

---

## 7. Key Implementation Details

### 7.1 Batching Strategy

The project uses a sophisticated batching mechanism:

1. **Snapshot-aware Batching:**
   - Each batch contains multiple temporal snapshots for the same nodes
   - Nodes may not exist in all snapshots
   - `exist_nodes` tracks which nodes are active in each snapshot

2. **Data Caching:**
   - Computed batches are cached to `./data/{dataset}/final_data/` to avoid recomputation
   - Cache key: `batch-size-{batch_size}/seed-{seed}/{split_type}/`

3. **Neighbor Sampling:**
   - Uses PyG's `NeighborLoader` for efficient sampling
   - Extracts k-hop neighborhoods for each node across all snapshots

### 7.2 Feature Fusion Strategy

The NodeFeatureEmbeddingLayer uses **equal-weighted fusion** of four feature types:
```
Output_dim = hidden_dim
Each feature → hidden_dim/4 (¼ of total)

Total = ¼ numerical + ¼ categorical + ¼ description + ¼ tweets
```

This ensures balanced contribution from all modalities.

### 7.3 Temporal Masking in Attention

The GraphTemporalLayer implements **causal attention**:
```
Attention Mask:
At time t, a node can only attend to times [0, 1, ..., t]
Cannot attend to future timesteps [t+1, t+2, ..., T-1]

This enforces temporal causality during training.
```

---

## 8. Configuration Parameters

**Default Configuration (Twibot-20):**
```python
--dataset_name: 'Twibot-20'
--seed: 1234
--device: 'cuda:0'
--interval: 'year'
--early_stop: False
--patience: 10
--coefficient: 1.1 (temporal loss weighting)
--temporal_head_config: 4
--structural_head_config: 4
--batch_size: 64
--hidden_dim: 128
--temporal_drop: 0.5
--structural_drop: 0.0
--structural_learning_rate: 1e-4
--temporal_learning_rate: 1e-5
--weight_decay: 1e-2
--epoch: 20
--window_size: -1 (use all snapshots)
--temporal_module_type: 'attention'
```

**For Twibot-22:**
```python
--batch_size: 256
--hidden_dim: 64
--weight_decay: 5e-2
--structural_learning_rate: 5e-4
--temporal_learning_rate: 5e-5
```

---

## 9. Key Data Processing Steps (Preprocessing)

**File:** `data/Twibot-20/preprocess_twibot20.py`

1. **User Extraction:** Separate users from other node types in `node.json`
2. **Global Indexing:** Create mapping from user IDs to global indices
3. **Temporal Splitting:** Group users by account creation date into time windows
4. **Graph Snapshotting:** For each time window, create a graph containing all users created before that time
5. **Edge Filtering:** Filter edges to only include nodes in current snapshot
6. **Feature Extraction:** Extract network statistics (clustering coefficient, bidirectional links, etc.)

---

## 10. File Structure Summary

| Component | File | Purpose |
|-----------|------|---------|
| Entry Point | `main.py` | Training orchestration |
| Config | `config.py` | Argument parsing |
| Model Core | `models/model.py` | BotDyGNN main model |
| Feature Embedding | `models/NodeFeatureEmbeddingLayer.py` | Multi-modal feature fusion |
| Structural Processing | `models/GraphStructuralLayer.py` | Graph convolution layer |
| Temporal Processing | `models/GraphTemporalLayer.py` | Temporal attention layer |
| Position Encoding | `models/PositionEmbeddingLayer.py` | Topology-aware positional encodings |
| Dataset | `utils/dataset.py` | Data loading and batching |
| Loss | `utils/loss.py` | Temporal weighted loss |
| Metrics | `utils/metrics.py` | Evaluation metrics |
| Preprocessing | `data/Twibot-20/preprocess_twibot20.py` | Data preparation pipeline |

---

## 11. Critical Algorithm Logic

### 11.1 Forward Pass (Complete Pipeline)

```
Input:
  - Multi-modal features for T snapshots
  - Edge lists for T snapshots
  - Graph properties (clustering coefficient, bidirectional links)
  - Existence masks

Process:
  For each snapshot t:
    1. Embed features: x_t = NodeFeatureEmbedding(des_t, tweet_t, num_t, cat_t)
    2. Apply structural layer: g_t = GraphStructural(x_t, edge_t)
    3. Compute position encodings:
       - p_cc_t = PositionEncodingCC(clustering_coef_t)
       - p_bl_t = PositionEncodingBL(bidirectional_links_t)
  
  Stack all snapshots: G = [g_0, g_1, ..., g_{T-1}]
  
  Apply temporal layer:
    output = GraphTemporal(G, p_cc, p_bl, exist_nodes)
    
Return: Classification logits [B, T, 2]
```

### 11.2 Loss with Temporal Weighting

```python
total_loss = 0
for t in range(num_snapshots):
    if exist_nodes[t].any():
        snapshot_loss = CrossEntropy(output[t], label[t])
        weight = 1.1^t  # Exponential weight favoring recent snapshots
        total_loss += snapshot_loss * weight
```

### 11.3 Batch Iteration Pattern

```python
for batch_data in dataloader:
    # 1. Prepare batch data across all snapshots
    # 2. Move to device
    # 3. Forward pass: output = model(batch_data)
    # 4. Compute loss: loss = loss_fn(output, labels)
    # 5. Backward: loss.backward()
    # 6. Optimizer step
    # 7. Compute metrics on final snapshot
```

---

## 12. Performance Output Format

**Training Output File Example:**
```
output/Twibot-20/
└── year + 1234 + 0.87574 + 0.89071.pt
    ├── Format: {interval}+{seed}+{val_f1}+{test_f1}.pt
    └── Content: Model state dict (weights)
```

Metrics tracked during training:
- Training loss (per epoch)
- Validation accuracy, F1, precision, recall (per epoch)
- Test metrics at best validation checkpoint

---

## Summary

**BotDGT** is a sophisticated temporal graph neural network for bot detection that:

1. **Fuses multi-modal features** (text embeddings + structural properties) efficiently
2. **Processes structural dynamics** through graph transformers with multi-head attention
3. **Models temporal evolution** through attention mechanisms with causal masking
4. **Weights recent behavior** more heavily in training via exponential loss coefficients
5. **Leverages network topology** through position encodings (clustering coefficient, bidirectional links)
6. **Implements efficient batching** with snapshot-aware data loading and caching

The architecture is particularly effective because it captures both:
- **Spatial patterns:** How accounts interact with others (structural layer)
- **Temporal patterns:** How interaction patterns change over time (temporal layer)
- **Behavioral signals:** Text embeddings and account statistics (feature layer)

This multi-level analysis enables more accurate bot detection compared to methods that only use static snapshots or ignore text information.
