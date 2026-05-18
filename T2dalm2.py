from lattice import *
from scipy.linalg import eigh
import numpy as np
import time
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# ==========================================
# 1. 基礎自洽函數 
# ==========================================
def fermi(eps, beta):
    return np.where(beta * eps > 0,
                    np.exp(-beta * eps) / (1 + np.exp(-beta * eps)),
                    1 / (1 + np.exp(beta * eps)))

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
    return (n_target - ntot).real 

def compute_mu(mu0, ek_list, U, n1up, n1dn, n2up, n2dn, beta, field, n_target):
    sols = root_scalar(func_mu, bracket=(-50, 50), 
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
            print(f"    [SCF 收斂] 迭代次數: {it}, 總誤差: {diff:.2e}")
            break

    final_ns = [nup1, ndn1, nup2, ndn2]
    mx = nup1 - ndn1
    my = nup2 - ndn2
    m_alm = abs(mx - my) / 2.0

    return m_alm, final_ns, mu0

# === 獲取 t=0 時所有 k 點的初始密度矩陣 ===
def get_initial_rho_k(ek_list, U, ns, mu, beta):
    rho_list = []
    n1up, n1dn, n2up, n2dn = ns
    for hk in ek_list:
        H = hk.copy().astype(complex)
        H[0, 0] += U * n1dn - mu
        H[1, 1] += U * n1up - mu
        H[2, 2] += U * n2dn - mu
        H[3, 3] += U * n2up - mu
        rho = F_of_H(H, beta)
        rho_list.append(rho)
    return rho_list

if __name__ == "__main__":
    # ==========================================
    # 2. 設定你要測試的「單一參數點」 (初始狀態)
    # ==========================================
    U_initial = 1.0      
    delta_target = -0.15  
    beta = 50.            
    
    n_target = 2.0 + delta_target

    # ==========================================
    # 3. 建立晶格與動能矩陣
    # ==========================================
    lat = lattice(2,[[1,0,0],[0,1,0],[0,0,0]],[40,40,1],2,[[0,0,0],[0,0,0]])
    Na = 2
    t1 = 1.0
    t2 = 1.75
    t3 = 0.85
    t4 = 0.65
    
    hop = {}
    hop[0] = [[0,1,0,0,t1], [0,-1,0,0,t1], [0,0,1,0,t2], [0,0,-1,0,t2],
              [0,1,1,0,t3], [0,-1,-1,0,t3], [0,1,-1,0,t3], [0,-1,1,0,t3],
              [1,1,1,0,t4], [1,-1,-1,0,t4], [1,1,-1,0,-t4], [1,-1,1,0,-t4]]
    hop[1] = [[1,1,0,0,t2], [1,-1,0,0,t2], [1,0,1,0,t1], [1,0,-1,0,t1],
              [1,1,1,0,t3], [1,-1,-1,0,t3], [1,1,-1,0,t3], [1,-1,1,0,t3],
              [0,1,1,0,t4], [0,-1,-1,0,t4], [0,1,-1,0,-t4], [0,-1,1,0,-t4]]
            
    hop_mat = lat.build_mat_TB_PBC(hop, antiperiodic=False)
    hop_k = lat.FT_hop_ij()

    ek_all_list = []
    for i in np.arange(0, hop_k.shape[0], Na):
        tmp = hop_k[i:i+Na, i:i+Na]
        tmp = np.kron(tmp, np.eye(2))
        ek_all_list.append(tmp)

    Nk = len(ek_all_list)

    # ==========================================
    # 4. 執行 t <= 0 的靜態自洽計算
    # ==========================================
    print(f"\n=== 執行 t=0 初始態自洽計算 ===")
    print(f"初始參數: U_i = {U_initial}, delta = {delta_target}")
    
    n0 = n_target / 4.0
    guess_ALM = [n0 + 0.005, n0 - 0.005, n0 - 0.005, n0 + 0.005]

    D_alm_0, ns_alm_0, mu_final = run_scf(ek_all_list, U_initial, n_target, guess_ALM, beta)
    print(f"初始交替磁性序參數 D_alm(t=0) = {D_alm_0:.6f}")

    # ==========================================
    # 5. 時間演化 (Matrix Exponential Method)
    # ==========================================
    print("\n=== 開始時間演化 (Unitary Evolution Quench to U_f = 25.0) ===")
    U_final = 50.0
    
    # dt 的限制通常來自於物理震盪頻率而非數值發散
    dt = 0.001 
    t_max = 10.0  
    N_steps = int(t_max / dt)

    rho_k_list = get_initial_rho_k(ek_all_list, U_initial, ns_alm_0, mu_final, beta)
    current_ns = np.array(ns_alm_0)

    time_array = []
    alm_array = []
    
    start_time = time.time()

    for step in range(N_steps):
        t = step * dt
        time_array.append(t)
        
        # 記錄當下的 D_alm
        mx = current_ns[0] - current_ns[1]
        my = current_ns[2] - current_ns[3]
        m_alm = abs(mx - my) / 2.0
        alm_array.append(m_alm)
        
        next_rho_k_list = []
        n_sum = np.zeros(4, dtype=complex)
        
        for i, hk in enumerate(ek_all_list):
            rho_k = rho_k_list[i]
            
            # 構建當下的 H_MF(t)
            H_MF = hk.copy().astype(complex)
            H_MF[0, 0] += U_final * current_ns[1] - mu_final
            H_MF[1, 1] += U_final * current_ns[0] - mu_final
            H_MF[2, 2] += U_final * current_ns[3] - mu_final
            H_MF[3, 3] += U_final * current_ns[2] - mu_final
            
            # ----------------------------------------------------
            # 核心修改區塊： (矩陣指數法)
            # ----------------------------------------------------
            # 1. 對角化當下的哈密頓量: H = V * D * V^dagger
            evals, V = np.linalg.eigh(H_MF)
            
            # 2. 計算時間演化算符 U_evol = V * exp(-i * D * dt) * V^dagger
            # np.diag 將純量 exp 轉為對角矩陣
            exp_D = np.diag(np.exp(-1j * evals * dt))
            U_evol = V @ exp_D @ V.conj().T
            
            # 3. 推進密度矩陣: rho(t+dt) = U_evol * rho(t) * U_evol^dagger
            rho_k_next = U_evol @ rho_k @ U_evol.conj().T
            # ----------------------------------------------------
            
            next_rho_k_list.append(rho_k_next)
            
            # 收集下一時刻的對角線元素
            n_sum[0] += rho_k_next[0, 0]
            n_sum[1] += rho_k_next[1, 1]
            n_sum[2] += rho_k_next[2, 2]
            n_sum[3] += rho_k_next[3, 3]
            
        rho_k_list = next_rho_k_list
        current_ns = (n_sum / Nk).real 
        
        if step % (N_steps // 10) == 0:
            print(f"  演化進度: {t:.3f} / {t_max} | D_alm = {m_alm:.5f}")

    elapsed_time = time.time() - start_time
    print(f"時間演化完成！耗時 {elapsed_time:.2f} 秒")

    # ==========================================
    # 6. 繪製演化結果
    # ==========================================
    plt.figure(figsize=(8, 5))
    plt.plot(time_array, alm_array, label=f'Quench: U={U_initial} $\\rightarrow$ {U_final}\n(Matrix Exponential)', color='red', lw=2)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('Time Evolution of Altermagnetic Order', fontsize=14)
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel(r'$\Delta_{alm}(t)$', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()