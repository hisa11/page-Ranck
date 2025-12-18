import numpy as np
import matplotlib.pyplot as plt


def visualize_pagerank_convergence():
    # ---------------------------------------------------------
    # 1. グラフ定義（Bが人気者のネットワーク）
    # ---------------------------------------------------------
    # ノード: 0=A, 1=B, 2=C, 3=D
    # B に多くのリンクが集まる構造
    adjacency = np.array([
        [0, 1, 0, 0],  # A -> B
        [0, 0, 0, 0],  # B -> (ダングリングノード)
        [0, 1, 0, 0],  # C -> B
        [0, 1, 0, 0]   # D -> B
    ], dtype=float)

    N = adjacency.shape[0]
    alpha = 0.85      # ダンピングファクター
    max_iter = 20     # 反復回数

    # ---------------------------------------------------------
    # 2. 遷移確率行列 M の作成（ダングリング対応）
    # ---------------------------------------------------------
    row_sums = adjacency.sum(axis=1)
    M = np.zeros_like(adjacency)

    for i in range(N):
        if row_sums[i] == 0:
            # ダングリングノードは全ノードへ均等に分配
            M[i, :] = 1.0 / N
        else:
            M[i, :] = adjacency[i, :] / row_sums[i]

    # PageRank 用に転置（列確率行列）
    M = M.T

    # ---------------------------------------------------------
    # 3. PageRank 計算
    # ---------------------------------------------------------
    v = np.ones(N) / N     # 初期ランク
    history = [v.copy()]

    for _ in range(max_iter):
        v = alpha * (M @ v) + (1 - alpha) / N
        history.append(v.copy())

    history = np.array(history)

    # ---------------------------------------------------------
    # 4. 可視化
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    labels = ['Page A', 'Page B (King)', 'Page C', 'Page D']
    linestyles = ['--', '-', '--', '--']
    linewidths = [2, 4, 2, 2]

    for i in range(N):
        plt.plot(
            history[:, i],
            label=labels[i],
            color=colors[i],
            marker=markers[i],
            linestyle=linestyles[i],
            linewidth=linewidths[i],
            markersize=8,
            alpha=0.9
        )

    plt.title("PageRank Convergence Process", fontsize=18, fontweight='bold')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Rank Score", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='center right')

    # 収束目安ライン
    plt.axvline(x=10, color='gray', linestyle=':', alpha=0.8)
    plt.text(10.5, 0.38, "Convergence Area", fontsize=12, color='gray')

    plt.ylim(0, 1.0)
    plt.xlim(0, max_iter)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_pagerank_convergence()
