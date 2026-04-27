from lattice import *
from scipy.linalg import eigh
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar 

# ==========================================
# 1. 基礎函數 (SCF 迭代核心)
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
    return (n_target - (ntot / Nk)).real 

def compute_mu(mu0, ek_list, U, n1up, n1dn, n2up, n2dn, beta, field, n_target):
    sols = root_scalar(func_mu, bracket=(-30, 30), 
                       args=(ek_list, U, n1up, n1dn, n2up, n2dn, beta, field, n_target), 
                       method='bisect', xtol=1e-8)
    return sols.root

def run_scf(ek_list, U, n_target, inital_guess, beta, maxiter=500): 
    field = 0.00
    tol = 1e-8 
    mix = 0.5
    nup1, ndn1, nup2, ndn2 = inital_guess
    mu0 = (n_target / 4.0) * U

    for it in range(maxiter):
        current_ns = [nup1, ndn1, nup2, ndn2]
        new_ns = compute_densities(ek_list, U, current_ns, mu0, beta, field)
        n1up_new, n1dn_new, n2up_new, n2dn_new = new_ns
        
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
# 2. 能帶結構專用函數 (解析式與路徑)
# ==========================================
def get_H0_k(kx, ky):
    """依照論文 eq(1) & eq(2) 直接給出特定 k 點的 H0 (動能項)"""
    t1 = -1.0
    t2 = -1.75
    t3 = -0.85
    t4 = -0.65
    
    eps_x = -2*t1*np.cos(kx) - 2*t2*np.cos(ky) - 4*t3*np.cos(kx)*np.cos(ky)
    eps_y = -2*t2*np.cos(kx) - 2*t1*np.cos(ky) - 4*t3*np.cos(kx)*np.cos(ky)
    eps_xy = -4*t4*np.sin(kx)*np.sin(ky)
    
    # 2x2 軌域矩陣
    H_orb = np.array([
        [eps_x, eps_xy],
        [eps_xy, eps_y]
    ])
    # 擴展成 4x4 (包含自旋)
    return np.kron(H_orb, np.eye(2))

def get_k_path(nk_per_segment=100):
    G = np.array([0, 0])
    X = np.array([np.pi, 0])
    M = np.array([np.pi, np.pi])
    Y = np.array([0, np.pi])
    
    path_segments = [(G, X), (X, M), (M, Y), (Y, G), (G, M)]
    k_list = []
    tick_indices = [0]
    
    for start, end in path_segments:
        for i in range(nk_per_segment):
            k = start + (end - start) * (i / nk_per_segment)
            k_list.append(k)
        tick_indices.append(len(k_list))
    
    return np.array(k_list), tick_indices

# ==========================================
# 3. 主程式
# ==========================================
if __name__ == "__main__":
    # --- 步驟 A: 利用原版 Lattice 產生二維網格來做高精度 SCF ---
    print("Initializing Lattice Grid for SCF...")
    lat = lattice(2,[[1,0,0],[0,1,0],[0,0,0]],[40,40,1],2,[[0,0,0],[0,0,0]])
    Na = 2
    t1 = 1.0; t2 = 1.75; t3 = 0.85; t4 = 0.65
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
    
    hop_mat = lat.build_mat_TB_PBC(hop, antiperiodic=False)
    hop_k = lat.FT_hop_ij()

    ek_all_list = []
    for i in np.arange(0, hop_k.shape[0], Na):
        tmp = hop_k[i:i+Na, i:i+Na]
        ek_all_list.append(np.kron(tmp, np.eye(2)))

    # --- 步驟 B: 設定特定點參數並執行 SCF ---
    U_target = 20.0
    delta_target = -0.15

    n_target = 2.0 + delta_target
    beta = 200.0
    guess_ALM = [0.6 , 0.4, 0.4, 0.6] 

    print(f"Running SCF for U/t = {U_target}, delta = {delta_target} ...")
    start_time = time.time()
    m_alm, final_ns = run_scf(ek_all_list, U_target, n_target, guess_ALM, beta)
    mu_final = compute_mu(0.0, ek_all_list, U_target, *final_ns, beta, 0.0, n_target)
    
    print(f"SCF Done! Time: {time.time()-start_time:.2f}s")
    print(f"Altermagnetic order: {m_alm:.4f}")
    print(f"Densities: {final_ns}")
    print(f"Chemical Potential (mu): {mu_final:.4f}")

    # --- 步驟 C: 建立 k 路徑與計算能帶 ---
    print("Calculating bands along symmetry lines...")
    k_path, ticks = get_k_path(100)
    
    bands_up = []
    bands_dn = []
    
    for k in k_path:
        # 使用解析式產生 H0
        H_hf = get_H0_k(k[0], k[1])
        
        # 加入 HF 平均場項，並扣除化學勢 (讓 0 對齊費米能階)
        H_hf[0, 0] += U_target * final_ns[1] - mu_final 
        H_hf[1, 1] += U_target * final_ns[0] - mu_final 
        H_hf[2, 2] += U_target * final_ns[3] - mu_final 
        H_hf[3, 3] += U_target * final_ns[2] - mu_final 
        
        evals, evecs = np.linalg.eigh(H_hf)
        
        # 區分自旋向上與向下
        up_evals = []
        dn_evals = []
        for i in range(4):
            # 計算特徵向量在自旋基底的投影 Sz
            # 索引對應：0(orb1_up), 1(orb1_dn), 2(orb2_up), 3(orb2_dn)
            Sz = np.abs(evecs[0, i])**2 - np.abs(evecs[1, i])**2 + np.abs(evecs[2, i])**2 - np.abs(evecs[3, i])**2
            if Sz > 0:
                up_evals.append(evals[i])
            else:
                dn_evals.append(evals[i])
                
        # 確保排序正確
        bands_up.append(np.sort(up_evals))
        bands_dn.append(np.sort(dn_evals))
        
    bands_up = np.array(bands_up)
    bands_dn = np.array(bands_dn)

    # --- 步驟 D: 畫出如同論文 Fig. 3(a) 的圖 ---
    plt.figure(figsize=(6, 8))
    
    # 畫出自旋向上(紅)與自旋向下(藍)的能帶
    for i in range(2):
        plt.plot(bands_up[:, i], color='red', lw=1.5, alpha=0.8, label='Spin Up' if i==0 else "")
        plt.plot(bands_dn[:, i], color='royalblue', lw=1.5, alpha=0.8, label='Spin Down' if i==0 else "")

    plt.axhline(0, color='gray', linestyle='-', lw=0.5, alpha=0.5)
    
    # 標示高對稱點
    plt.xticks(ticks, [r'$\Gamma$', r'$X$', r'$M$', r'$Y$', r'$\Gamma$', r'$M$'], fontsize=14)
    for t in ticks:
        plt.axvline(t, color='gray', linestyle='--', lw=0.5, alpha=0.5)
        
    plt.ylabel(r'$\omega/t$', fontsize=16)
    plt.title('Hartree-Fock Band Structure', fontsize=16)
    plt.ylim(-15, 25)
    plt.xlim(0, max(ticks))
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# ==========================================
    # --- 步驟 E: 繪製 2D 費米面 (對應論文 inset 圖片) ---
    # ==========================================
    print("Calculating 2D Fermi Surface...")
    
    N_k = 150 
    # 🌟 關鍵修改：將網格平移到 0 ~ 2*pi，這樣中心點就是 (pi, pi) 也就是 M 點！
    kx_arr = np.linspace(-np.pi, np.pi, N_k)
    ky_arr = np.linspace(-np.pi, np.pi, N_k)
    Kx, Ky = np.meshgrid(kx_arr, ky_arr)
    
    # 準備存放 4 條能帶能量的陣列
    E_up_lower = np.zeros((N_k, N_k))
    E_up_upper = np.zeros((N_k, N_k))
    E_dn_lower = np.zeros((N_k, N_k))
    E_dn_upper = np.zeros((N_k, N_k))

    # 2. 遍歷網格計算能量
    for i in range(N_k):
        for j in range(N_k):
            H_hf = get_H0_k(Kx[i, j], Ky[i, j])
            
            # 加入 HF 平均場項 (使用先前的 final_ns 與 mu_final，並包含 -0.5 修正)
            H_hf[0, 0] += U_target * final_ns[1] - mu_final 
            H_hf[1, 1] += U_target * final_ns[0] - mu_final 
            H_hf[2, 2] += U_target * final_ns[3] - mu_final 
            H_hf[3, 3] += U_target * final_ns[2] - mu_final 
            
            evals, evecs = np.linalg.eigh(H_hf)
            
            # 區分自旋
            up_evals = []
            dn_evals = []
            for n in range(4):
                Sz = np.abs(evecs[0, n])**2 - np.abs(evecs[1, n])**2 + np.abs(evecs[2, n])**2 - np.abs(evecs[3, n])**2
                if Sz > 0:
                    up_evals.append(evals[n])
                else:
                    dn_evals.append(evals[n])
            
            E_up_lower[i, j] = up_evals[0]
            E_up_upper[i, j] = up_evals[1]
            E_dn_lower[i, j] = dn_evals[0]
            E_dn_upper[i, j] = dn_evals[1]

    # 3. 開始繪圖
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 設定背景顏色 (對應論文的灰色背景)
    ax.set_facecolor('darkgray')

    ax.contourf(Kx, Ky, E_up_lower, levels=[0, 100], colors=['white'], alpha=0.6)
    ax.contourf(Kx, Ky, E_dn_lower, levels=[0, 100], colors=['white'], alpha=0.6)
    # 畫出能量等於 0 (費米能階) 的等高線
    # 紅色：自旋向上
    ax.contour(Kx, Ky, E_up_lower, levels=[0], colors='tomato', linewidths=2)
    ax.contour(Kx, Ky, E_up_upper, levels=[0], colors='tomato', linewidths=2)
    # 藍色：自旋向下
    ax.contour(Kx, Ky, E_dn_lower, levels=[0], colors='cornflowerblue', linewidths=2)
    ax.contour(Kx, Ky, E_dn_upper, levels=[0], colors='cornflowerblue', linewidths=2)

    ax.plot([0], [0], 'ko', markersize=6)                 # Gamma (現在在正中央)
    ax.plot([np.pi, -np.pi], [0, 0], 'ko', markersize=6)  # X (左右邊緣中間)
    ax.plot([0, 0], [np.pi, -np.pi], 'ko', markersize=6)  # Y (上下邊緣中間)
    ax.plot([np.pi, np.pi, -np.pi, -np.pi], 
            [np.pi, -np.pi, np.pi, -np.pi], 'ko', markersize=6) # M (四個角落)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    
    ax.set_aspect('equal') 
    ax.set_xticks([])      
    ax.set_yticks([])      

    plt.tight_layout()
    plt.savefig("Fermi_Surface_Centered_Gamma.png", dpi=300)
    plt.show()
