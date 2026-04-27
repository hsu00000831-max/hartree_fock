from lattice import *
from scipy.linalg import eigh
import numpy as np
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # 多核心套件
from tqdm import tqdm # 進度條套件

# ==========================================
# 1. 基礎函數 (已修正全域變數 n 的問題)
# ==========================================
def fermi(eps, beta):
    return np.where(
        beta * eps > 0,
        np.exp(-beta * eps) / (1 + np.exp(-beta * eps)),
        1 / (1 + np.exp(beta * eps))
    )

def F_of_H(H, beta):
    eps, U_mat = np.linalg.eigh(H)
    f = fermi(eps, beta)
    return (U_mat * f) @ U_mat.conj().T

def compute_densities(ek_list, U, ns, mu, beta, field):
    Nk = len(ek_list)
    n1up_in ,n1dn_in ,n2up_in ,n2dn_in = ns 
    n1up_sum, n1dn_sum, n2up_sum, n2dn_sum = 0.0, 0.0, 0.0, 0.0
    for hk in ek_list:
        H = hk.copy()
        H[0, 0] += U * n1dn_in - field - mu
        H[1, 1] += U * n1up_in + field - mu
        H[2, 2] += U * n2dn_in - field - mu
        H[3, 3] += U * n2up_in + field - mu
        rho = F_of_H(H, beta)
        n1up_sum += rho[0, 0].real
        n1dn_sum += rho[1, 1].real
        n2up_sum += rho[2, 2].real
        n2dn_sum += rho[3, 3].real
    return np.array([n1up_sum, n1dn_sum, n2up_sum, n2dn_sum]) / Nk

def func_mu(x, ek_list, U, n1up, n1dn, n2up, n2dn, beta, field, n_target):
    mu = x
    ntot = 0.0
    Nk = len(ek_list)
    for hk in ek_list:
        H = hk.copy()
        H[0, 0] += U * n1dn - field - mu
        H[1, 1] += U * n1up + field - mu
        H[2, 2] += U * n2dn - field - mu
        H[3, 3] += U * n2up + field - mu
        evals = np.linalg.eigvalsh(H)
        ntot += np.sum(fermi(evals, beta))
    ntot = ntot/Nk
    # 這裡現在使用傳進來的 n_target，而不是全域變數 n
    return (n_target - ntot).real 

# --- 修正重點 2: compute_mu 必須傳遞 n_target ---
def compute_mu(mu0, ek_list, U, n1up, n1dn, n2up, n2dn, beta, field, n_target):
    from scipy.optimize import root_scalar 
    # args 順序必須跟 func_mu 定義的一樣 (x 後面的那些)
    sols = root_scalar(func_mu, bracket=(-20, 20), 
                           args=(ek_list, U, n1up, n1dn, n2up, n2dn, beta, field, n_target), 
                           method='bisect', xtol=1e-8)
    mu = sols.root
    
    return mu

#def get_free_energy(ek_list, U, ns, mu, beta, field, n_target):
    Nk = len(ek_list)
    n1up, n1dn, n2up, n2dn = ns
    term1 = 0.0
    for hk in ek_list:
        H = hk.copy()
        H[0, 0] += U * n1dn - field - mu
        H[1, 1] += U * n1up + field - mu
        H[2, 2] += U * n2dn - field - mu
        H[3, 3] += U * n2up + field - mu
        evals = np.linalg.eigvalsh(H)
        val = -beta * evals
        term1 += np.sum(np.logaddexp(0, val))
    Omega = - (1.0/beta) * term1 / Nk
    double_counting = U * (n1up * n1dn + n2up * n2dn)
    # 修正: 加上 mu * n 項才是 Grand Potential 轉 Free Energy 的正確操作
    return Omega - double_counting + mu * n_target

def run_scf(ek_list, U, n_target, inital_guess, beta, maxiter=500): 
    field = 0.00
    tol = 1e-8 
    mix = 0.5
    nup1, ndn1, nup2, ndn2 = inital_guess
    
    # 智慧型初始猜測
    mu0 = (n_target / 4.0) * U

    for it in range(maxiter):
        current_ns = [nup1, ndn1, nup2, ndn2]
        new_ns = compute_densities(ek_list, U, current_ns, mu0, beta, field)
        n1up_new, n1dn_new, n2up_new, n2dn_new = new_ns
        
        # 傳遞 n_target
        mu_new = compute_mu(mu0, ek_list, U, n1up_new, n1dn_new, n2up_new, n2dn_new, beta, field, n_target)

        diff = np.abs(mu_new - mu0) + np.sum(np.abs(new_ns - current_ns))
        
        nup1 = mix * n1up_new + (1 - mix) * nup1
        ndn1 = mix * n1dn_new + (1 - mix) * ndn1
        nup2 = mix * n2up_new + (1 - mix) * nup2
        ndn2 = mix * n2dn_new + (1 - mix) * ndn2
        mu0 = mix * mu_new + (1 - mix) * mu0
        
        if diff < tol:
            break

    final_ns = [nup1, ndn1, nup2, ndn2]
    mx = nup1 - ndn1
    my = nup2 - ndn2
    m_alm = abs(mx - my) / 2.0

    return m_alm, final_ns

# ==========================================
# 2. 並行運算的核心單元 (Solver)
# ==========================================
def solve_single_col(n_val, U_values, ek_all_list, beta):
    """
    這個函式負責計算「一整行」的相圖數據 (固定 U，掃描所有 n)。
    這樣分配任務比計算單點更有效率。
    """
    print(f"--> Core working on n = {n_val:.2f} ...", flush=True)
    col_results = []
    n_target = 2.0 + n_val
    guess_ALM = [0.8, 0.2, 0.2, 0.8]

    for U_val in U_values:
        #計算alm
        D_alm, ns_alm = run_scf(ek_all_list, U_val, n_target, guess_ALM, beta)

        # 濾除數值誤差
        if D_alm < 1e-3:
            D_alm = 0.0
            guess_ALM = [0.8, 0.2, 0.2, 0.8] # 磁性消失時，重置猜測值
        else:
            guess_ALM = ns_alm # 將收斂的結果作為下一個 U 的猜測值
            
        col_results.append(D_alm)

        print(f"    [n={n_val:5.2f}] finished U={U_val:4.2f}  ", flush=True)
        
    
    return col_results

# ==========================================
# 3. 主程式
# ==========================================
if __name__ == "__main__":
    # --- Lattice Setup ---
    # 建議網格加密一點，既然我們有多核心加速
    lat = lattice(2,[[1,0,0],[0,1,0],[0,0,0]],[60,60,1],2,[[0,0,0],[0,0,0]])
    Na = 2
    t1 = 1.0
    t2 = 1.75
    t3 = 0.85
    t4 = 0.65
    hop = {}
    hop[0] = [[0,1,0,0,t1],
        [0,-1,0,0,t1],
        [0,0,1,0,t2],
        [0,0,-1,0,t2],
        [0,1,1,0,t3],
        [0,-1,-1,0,t3],
        [0,1,-1,0,t3],
        [0,-1,1,0,t3],
        [1,1,1,0,t4],
        [1,-1,-1,0,t4],
        [1,1,-1,0,-t4],
        [1,-1,1,0,-t4]
        ]
    hop[1] = [[1,1,0,0,t2],
        [1,-1,0,0,t2],
        [1,0,1,0,t1],
        [1,0,-1,0,t1],
        [1,1,1,0,t3],
        [1,-1,-1,0,t3],
        [1,1,-1,0,t3],
        [1,-1,1,0,t3],
        [0,1,1,0,t4],
        [0,-1,-1,0,t4],
        [0,1,-1,0,-t4],
        [0,-1,1,0,-t4]
        ]
    hop_mat = lat.build_mat_TB_PBC(hop,antiperiodic=False)
    hop_k = lat.FT_hop_ij()

    ek_all_list = []
    for i in np.arange(0, hop_k.shape[0], Na):
        tmp = hop_k[i:i+Na, i:i+Na]
        tmp = np.kron(tmp, np.eye(2))
        ek_all_list.append(tmp)

    # --- Parameters ---
    beta = 50. 
    N_points_n = 101
    N_points_U = 97

    n_values = np.linspace(-2, 0.0, N_points_n) 
    U_values = np.linspace(25.0, 1.0, N_points_U)

    print(f"=== Starting Parallel Phase Diagram Scan ===")
    print(f"Grid: {len(n_values)}x{len(U_values)} = {len(n_values)*len(U_values)} points")
    print("Using all available CPU cores...")

    start_time = time.time()

    # --- 核心並行運算部分 ---
    # n_jobs=-1 代表使用所有 CPU 核心
    # delayed(solve_single_row)(...) 會把任務分配出去
    # tqdm 用來顯示進度條
    results = Parallel(n_jobs=-1)(
        delayed(solve_single_col)(n_val, U_values, ek_all_list, beta) 
        for n_val in tqdm(n_values, desc="Scanning cols(n)")
    )

    # 將結果轉回 numpy array
    phase_map = np.array(results).T

    elapsed_time = time.time() - start_time
    print(f"=== Scan Done! Total time: {elapsed_time:.2f} s ===")
    # 畫圖前將 U 的陣列與相圖矩陣「上下翻轉」，讓 Y 軸從小到大正常顯示
    U_values_plot = U_values[::-1]
    phase_map_plot = phase_map[::-1, :]  # 翻轉列 (U 的維度)
# --- Plotting ---
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # 建立網格
    delta_grid, U_grid = np.meshgrid(n_values, U_values_plot)
    
    # 畫熱圖
    c = ax.pcolormesh(delta_grid, U_grid, phase_map_plot, cmap='hot_r', shading='nearest', vmin=0, vmax=1.0)
    
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label(r'$\Delta_{alm}$', fontsize=16, rotation=0, labelpad=20)
    
    # 標示 vHs 虛線 
    ax.axvline(-0.145, color='gray', linestyle='--', lw=3, alpha=0.8)
    ax.axvline(-0.396, color='cornflowerblue', linestyle='--', lw=3, alpha=0.8, label=r'vHs ($\delta \approx -0.396$)')

    ax.set_xlabel(r'$\delta$', fontsize=16)
    ax.set_ylabel(r'$U/t$', fontsize=16)
    
    # ==========================================
    # 🌟 關鍵修改：讓座標軸自動適應你設定的陣列範圍
    # ==========================================
    ax.set_xlim(np.min(n_values), np.max(n_values))
    ax.set_ylim(np.min(U_values), np.max(U_values))
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig("ALM_Heatmap.png", dpi=300)
    print("圖檔已儲存為 ALM_Heatmap.png")