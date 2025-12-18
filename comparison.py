import numpy as np
import time

def benchmark_pagerank_step():
    print("=== PageRank実装対決：Standard vs NumPy ===")
    
    # 1. 設定とデータの準備
    N = 1000          # ノード数（この数を増やすと差が広がります）
    alpha = 0.85      # ダンピングファクター
    
    print(f"ノード数: {N}")
    print("テストデータ生成中...", end="")
    # テスト用に密行列と初期ベクトルを生成
    # (実際はスパースですが、計算負荷比較のため密行列でテストします)
    M_list = [[1.0/N] * N for _ in range(N)] # 標準リスト
    v_list = [1.0/N] * N
    
    M_np = np.array(M_list)                  # NumPy配列
    v_np = np.array(v_list)
    print("完了！\n")

    # ============================================
    # Round 1: Standard Python (List + Loop)
    # ============================================
    print("Round 1: Standard Python (計算中...)")
    start_std = time.time()
    
    v_next_std = [0.0] * N
    # --- ボトルネック箇所 ---
    for i in range(N):
        dot_val = 0.0
        for j in range(N):
            dot_val += M_list[i][j] * v_list[j]
        v_next_std[i] = alpha * dot_val + (1 - alpha) / N
    # ------------------------
    
    time_std = time.time() - start_std
    print(f"-> 完了タイム: {time_std:.6f} 秒\n")

    # ============================================
    # Round 2: NumPy (Vectorization)
    # ============================================
    print("Round 2: NumPy (計算中...)")
    start_np = time.time()
    
    # --- 高速化箇所 ---
    # np.dot によるベクトル化演算
    dot_val_np = np.dot(M_np, v_np)
    v_next_np = alpha * dot_val_np + (1 - alpha) / N
    # ------------------
    
    time_np = time.time() - start_np
    print(f"-> 完了タイム: {time_np:.6f} 秒\n")

    # ============================================
    # 結果発表
    # ============================================
    print("=== 結果発表 ===")
    speedup = time_std / time_np
    print(f"Standard Python : {time_std:.6f} 秒")
    print(f"NumPy           : {time_np:.6f} 秒")
    print(f"\n勝者: NumPy！ (約 {speedup:.1f} 倍高速)")

    # (念のため計算結果が同じか確認)
    diff = np.linalg.norm(np.array(v_next_std) - v_next_np)
    print(f"\n(計算結果の差分ノルム: {diff:.6e})")
    if diff < 1e-9:
        print("※計算結果はほぼ同一です。")

if __name__ == "__main__":
    benchmark_pagerank_step()
