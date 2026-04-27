from lattice import *
from scipy.linalg import eigh
import numpy as np
import time
from scipy.optimize import root_scalar

# ==========================================
# 1. 基礎自洽函數 (完美運作的核心邏輯)
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
        rho = F_of_H(H, beta).T
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

if __name__ == "__main__":
    # ==========================================
    # 2. 設定你要測試的「單一參數點」
    # ==========================================
    U_target = 20.0       # <--- 在這裡修改你要測試的 U
    delta_target = -0.15  # <--- 在這裡修改你要測試的摻雜濃度 delta
    beta = 50.            # 溫度參數
    
    n_target = 2.0 + delta_target

    # ==========================================
    # 3. 使用原來的 lattice 與 hop 字典
    # ==========================================
    # 為了單點計算，網格可以開到 60x60，速度很快又精準
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
            
    hop_mat = lat.build_mat_TB_PBC(hop, antiperiodic=False)
    hop_k = lat.FT_hop_ij()

    ek_all_list = []
    for i in np.arange(0, hop_k.shape[0], Na):
        tmp = hop_k[i:i+Na, i:i+Na]
        tmp = np.kron(tmp, np.eye(2))
        ek_all_list.append(tmp)

    # ==========================================
    # 4. 執行自洽計算
    # ==========================================
    print(f"\n=== 執行單點自洽計算 ===")
    print(f"目標參數: U = {U_target}, delta = {delta_target} (n = {n_target})")
    
    start_time = time.time()
    
    # 給予微小擾動的溫和初始猜測，引導 ALM 收斂
    n0 = n_target / 4.0
    guess_ALM = [n0 + 0.005, n0 - 0.005, n0 - 0.005, n0 + 0.005]

    D_alm, ns_alm, mu_final = run_scf(ek_all_list, U_target, n_target, guess_ALM, beta)
    
    elapsed_time = time.time() - start_time

    # ==========================================
    # 5. 輸出詳細結果
    # ==========================================
    print(f"\n=== 計算完成 (耗時 {elapsed_time:.2f} 秒) ===")
    print(f"最終化學勢 mu = {mu_final:.6f}")
    print(f"軌道 x 佔據數: n_x_up = {ns_alm[0]:.5f}, n_x_dn = {ns_alm[1]:.5f}")
    print(f"軌道 y 佔據數: n_y_up = {ns_alm[2]:.5f}, n_y_dn = {ns_alm[3]:.5f}")
    print("-" * 35)
    print(f"交替磁性序參數 D_alm = {D_alm:.6f}")