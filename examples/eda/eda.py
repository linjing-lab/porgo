'''
Estimation of Distribution Algorithm for Binary Optimization with PyTorch.
'''

from dataclasses import dataclass
from typing import Tuple, List, Callable, Dict
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 若可用则使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32  # 统一使用 float32；如需 bfloat16 可改为 torch.bfloat16


@dataclass(frozen=True)
class EDAConfig:
    dimensions: int = 128
    population_size: int = 100
    selection_ratio: float = 0.5
    learning_rate: float = 0.1
    epsilon: float = 1e-3
    max_evaluations: int = 100000
    print_every: int = 100
    dtype: torch.dtype = DTYPE
    eval_fn: Callable = None
    seed: int = 42


def create_initial_distribution(config: EDAConfig) -> torch.Tensor:
    """初始化概率分布，形状为 (dimensions,)"""
    return torch.full((config.dimensions,), 0.5, dtype=config.dtype, device=device)


def sample_population(
    distribution: torch.Tensor, config: EDAConfig, generator: torch.Generator
) -> torch.Tensor:
    """从当前概率分布采样种群，形状为 (population_size, dimensions)"""
    uniform = torch.rand(
        config.population_size, config.dimensions, dtype=config.dtype, device=device, generator=generator
    )
    return (uniform < distribution).to(config.dtype)


def evaluate_population(population: torch.Tensor, eval_fn: Callable) -> torch.Tensor:
    """评估种群，返回适应度数组，形状为 (population_size,)"""
    # 若 eval_fn 不支持批量，可在此处改为逐个体评估
    return torch.stack([eval_fn(ind) for ind in population], dim=0)


def select_elite(
    population: torch.Tensor, fitness: torch.Tensor, config: EDAConfig
) -> torch.Tensor:
    """选择精英解"""
    num_elite = max(1, int(round(config.population_size * config.selection_ratio)))
    elite_indices = torch.topk(fitness, k=num_elite, largest=True).indices
    return population[elite_indices]


def update_distribution(
    distribution: torch.Tensor, elite_solutions: torch.Tensor, config: EDAConfig
) -> torch.Tensor:
    """基于精英解更新概率分布"""
    empirical_probs = elite_solutions.mean(dim=0)
    new_distribution = (1.0 - config.learning_rate) * distribution + config.learning_rate * empirical_probs
    return torch.clamp(new_distribution, config.epsilon, 1.0 - config.epsilon)


def eda_step(
    distribution: torch.Tensor, config: EDAConfig, generator: torch.Generator
) -> Tuple[torch.Tensor, Dict, int]:
    """完成一代 EDA 步骤"""
    # 1) 采样
    population = sample_population(distribution, config, generator)

    # 2) 评估
    fitness = evaluate_population(population, config.eval_fn)
    eval_count = config.population_size

    # 3) 选择精英
    elite = select_elite(population, fitness, config)

    # 4) 更新分布
    new_distribution = update_distribution(distribution, elite, config)

    # 5) 指标
    # 避免 log(0)
    eps = 1e-10
    entropy = -torch.sum(
        distribution * torch.log(distribution + eps)
        + (1.0 - distribution) * torch.log(1.0 - distribution + eps)
    )

    metrics = {
        "fitness": fitness,
        "best_fitness": torch.max(fitness).item(),
        "mean_fitness": torch.mean(fitness).item(),
        "distribution_entropy": entropy.item(),
    }
    return new_distribution, metrics, eval_count


def run_eda(config: EDAConfig) -> Tuple[torch.Tensor, List[float]]:
    """运行完整的 EDA 优化"""
    generator = torch.Generator(device=device).manual_seed(config.seed)

    print("Starting Estimation of Distribution Algorithm (EDA)...")
    print(f"配置参数: {config}")
    print(f"问题维度: {config.dimensions}")
    print(f"种群大小: {config.population_size}")
    print(f"选择比例: {config.selection_ratio}")
    print(f"最大评估次数: {config.max_evaluations}")
    print("-" * 50)

    distribution = create_initial_distribution(config)
    fitness_history = []
    total_evaluations = 0
    generation = 0

    with tqdm(total=config.max_evaluations, desc="EDA Optimizing") as pbar:
        while total_evaluations < config.max_evaluations:
            distribution, metrics, eval_count = eda_step(distribution, config, generator)

            total_evaluations += eval_count
            generation += 1

            best_fitness = metrics["best_fitness"]
            mean_fitness = metrics["mean_fitness"]
            entropy = metrics["distribution_entropy"]

            fitness_history.append(best_fitness)

            pbar.update(eval_count)
            pbar.set_postfix(
                {
                    "Gen": generation,
                    "Best": f"{best_fitness:.4f}",
                    "Mean": f"{mean_fitness:.4f}",
                    "Entropy": f"{entropy:.2f}",
                }
            )

            if generation % config.print_every == 0 or total_evaluations >= config.max_evaluations:
                print(
                    f"代数 {generation:4d}, 评估次数 {total_evaluations:6d}: "
                    f"最佳适应度={best_fitness:.4f}, 平均适应度={mean_fitness:.4f}, 分布熵={entropy:.4f}"
                )

    print("-" * 50)
    print("EDA 优化完成!")
    print(f"总代数: {generation}")
    print(f"总评估次数: {total_evaluations}")
    print(f"最终最佳适应度: {fitness_history[-1]:.4f}")

    return distribution, fitness_history


if __name__ == "__main__":

    D = 1000
    torch.manual_seed(0)

    from eval_fns import one_max, hamming_distance, weighted_one_max, deceptive_trap, rugged_plateau

    config = EDAConfig(
        dimensions=D,
        eval_fn=hamming_distance,
        population_size=100,
        max_evaluations=20000,
        print_every=50,
        seed=42,
    )

    final_distribution, fitness_history = run_eda(config)

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    x_evals = [(i + 1) * config.population_size for i in range(len(fitness_history))]
    plt.plot(x_evals, fitness_history, linewidth=2, color="red", label="EDA")
    plt.title("EDA Convergence Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Function Evaluations", fontsize=12)
    plt.ylabel("Best Fitness", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("examples/figures/hamming_distance.png", dpi=300, bbox_inches="tight")
    plt.show()