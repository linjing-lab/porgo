import torch

def one_max(x: torch.Tensor) -> torch.Tensor:
    # 示例：最大化 x 中 1 的比例，并加入轻微正则避免全 1
    return torch.sum(x) - 1e-3 * torch.sum(x * x)

def hamming_distance(x: torch.Tensor) -> torch.Tensor:
    """最大化与给定中心向量的汉明距离"""
    center = torch.zeros_like(x)
    return (x != center).sum(dim=-1).float()

def weighted_one_max(x: torch.Tensor) -> torch.Tensor:
    """最大化加权和 w·x；w.shape == x.shape[-1]"""
    w = torch.linspace(1.0, 2.0, 1000).to(device=x.device)
    return (x * w).sum(dim=-1).float()

def deceptive_trap(x: torch.Tensor) -> torch.Tensor:
    """
    欺骗陷阱函数（Deceptive Trap）。
    将每 k 位划分为一块，块内全1得 k，否则得 k-1-该块中0的个数。
    最大化所有块的得分之和。
    """
    # 形状: (..., n_blocks, k)
    blocks = x.reshape(*x.shape[:-1], -1, 4)
    ones_per_block = blocks.sum(dim=-1)  # (..., n_blocks)
    block_scores = torch.where(
        ones_per_block == 4,
        torch.tensor(4, dtype=x.dtype, device=x.device),
        4 - 1 - ones_per_block
    )
    return block_scores.sum(dim=-1).float()

def rugged_plateau(x: torch.Tensor, plateau_val: float = 2.0) -> torch.Tensor:
    """
    崎岖高原函数：在每个 plateau_len 长度的窗口内，若全1则得 plateau_val，
    否则得窗口内1的个数。最大化总得分。
    """
    wins = x.reshape(*x.shape[:-1], -1, 5)
    ones = wins.sum(dim=-1)
    scores = torch.where(ones == 5, plateau_val, ones)
    return scores.sum(dim=-1).float()