import numpy as np
import matplotlib.pyplot as plt
f_c = 1.8e9
f_c_ghz = 1.8
c = 3.0e8
lambda_0 = c / f_c
N_BS_col = 4
N_BS_pol = 2
S = N_BS_col * N_BS_pol
U = 2
d_H = 0.5 * lambda_0
d_V = 0.5 * lambda_0
h_BS = 25.0
h_UT = 1.5
bs_pos = np.array([0.0, 0.0, h_BS])
ut_pos = np.array([500.0, 300.0, h_UT])
d_2D = np.linalg.norm(ut_pos[:2] - bs_pos[:2])
d_3D = np.linalg.norm(ut_pos - bs_pos)
delta = ut_pos - bs_pos
phi_LOS_AOD = np.degrees(np.arctan2(delta[1], delta[0]))
phi_LOS_AOA = phi_LOS_AOD + 180.0
theta_LOS_ZOD = np.degrees(np.arccos(delta[2] / d_3D))
theta_LOS_ZOA = 180.0 - theta_LOS_ZOD
d_tx = np.zeros((S, 3))
for col in range(N_BS_col):
    d_tx[col * 2, :] = [col * d_H, 0.0, 0.0]
    d_tx[col * 2 + 1, :] = [col * d_H, 0.0, d_V]
d_rx = np.zeros((U, 3))
d_rx[0, :] = [0.0, 0.0, 0.0]
d_rx[1, :] = [d_H, 0.0, 0.0]
print("STEP 1: Environment setup")
print(f"  Scenario: UMa NLOS | f_c = {f_c_ghz} GHz | lambda = {lambda_0*100:.2f} cm")
print(f"  BS: {S} elements (4-col x 2-pol) | UT: {U} elements")
print(f"  d_2D = {d_2D:.1f} m | d_3D = {d_3D:.1f} m")
print(f"  LOS AOD = {phi_LOS_AOD:.1f}° | LOS AOA = {phi_LOS_AOA:.1f}°")
print(f"  LOS ZOD = {theta_LOS_ZOD:.1f}° | LOS ZOA = {theta_LOS_ZOA:.1f}°")
propagation = "NLOS"
print(f"\nSTEP 2: Propagation condition = {propagation}")
PL_uma_nlos = 13.54 + 39.08 * np.log10(d_3D) + 20 * np.log10(f_c_ghz) - 0.6 * (h_UT - 1.5)
sigma_SF = 6.0
SF = np.random.normal(0, sigma_SF)
print(f"\nSTEP 3: Path loss & shadowing")
print(f"  PL_UMa_NLOS = {PL_uma_nlos:.2f} dB | SF = {SF:.2f} dB")
fc_lsp = max(f_c_ghz, 6.0)
mu_lgDS  = -6.28 - 0.204 * np.log10(fc_lsp)
sig_lgDS = 0.39
mu_lgASD = 1.5 - 0.1144 * np.log10(fc_lsp)
sig_lgASD = 0.28
mu_lgASA = 2.08 - 0.27 * np.log10(fc_lsp)
sig_lgASA = 0.11
mu_lgZSA = -0.3236 * np.log10(fc_lsp) + 1.512
sig_lgZSA = 0.16
d_2D_km = d_2D / 1000.0
mu_lgZSD = max(-0.5, -2.1 * d_2D_km - 0.01 * (h_UT - 1.5) + 0.9)
sig_lgZSD = 0.49
C = np.array([
    [ 1.0,   0.4,   0.6,   0.0,  -0.5],
    [ 0.4,   1.0,   0.4,  -0.1,   0.5],
    [ 0.6,   0.4,   1.0,   0.0,   0.0],
    [ 0.0,  -0.1,   0.0,   1.0,   0.0],
    [-0.5,   0.5,   0.0,   0.0,   1.0],
])
L_chol = np.linalg.cholesky(C)
z = np.random.randn(5)
s_corr = L_chol @ z
DS  = 10**(mu_lgDS  + sig_lgDS  * s_corr[0])
ASD = min(10**(mu_lgASD + sig_lgASD * s_corr[1]), 104.0)
ASA = min(10**(mu_lgASA + sig_lgASA * s_corr[2]), 104.0)
ZSA = min(10**(mu_lgZSA + sig_lgZSA * s_corr[3]), 52.0)
ZSD = min(10**(mu_lgZSD + sig_lgZSD * s_corr[4]), 52.0)
print(f"\nSTEP 4: Large scale parameters (LSPs)")
print(f"  DS  = {DS*1e9:.2f} ns | ASD = {ASD:.2f}° | ASA = {ASA:.2f}°")
print(f"  ZSA = {ZSA:.2f}° | ZSD = {ZSD:.2f}°")
N_clusters = 20
M_rays = 20
r_tau = 2.3
X_n = np.random.uniform(0, 1, N_clusters)
tau_prime = -r_tau * DS * np.log(X_n)
tau = np.sort(tau_prime - np.min(tau_prime))
print(f"\nSTEP 5: Cluster delays")
print(f"  tau_min = {tau[0]*1e9:.2f} ns | tau_max = {tau[-1]*1e9:.2f} ns")
print(f"  RMS delay spread (input) = {DS*1e9:.2f} ns")
zeta = 3.0
Z_n = np.random.normal(0, zeta, N_clusters)
P_prime = np.exp(-tau * (r_tau - 1) / (r_tau * DS)) * 10**(-Z_n / 10.0)
P_n = P_prime / np.sum(P_prime)
P_max = np.max(P_n)
keep = P_n >= P_max * 10**(-25.0/10.0)
tau = tau[keep]
P_n = P_n[keep]
P_n = P_n / np.sum(P_n)
N_active = len(tau)
print(f"\nSTEP 6: Cluster powers")
print(f"  Active clusters (after -25dB pruning): {N_active}/{N_clusters}")
print(f"  Strongest cluster power: {10*np.log10(P_n[0]):.1f} dB (relative)")
alpha_m = np.array([
    0.0447, -0.0447, 0.1413, -0.1413,
    0.2492, -0.2492, 0.3715, -0.3715,
    0.5129, -0.5129, 0.6797, -0.6797,
    0.8844, -0.8844, 1.1481, -1.1481,
    1.5195, -1.5195, 2.1551, -2.1551
])
c_ASD = 2.0
c_ASA = 15.0
c_ZSA = 7.0
C_phi_NLOS = 1.289
C_theta_NLOS = 1.178
phi_AOA_prime = 2.0 * (ASA / 1.4) * np.sqrt(-np.log(P_n / np.max(P_n))) / C_phi_NLOS
X_n_aoa = np.random.choice([-1, 1], N_active)
Y_n_aoa = np.random.normal(0, ASA / 7.0, N_active)
phi_n_AOA = X_n_aoa * phi_AOA_prime + Y_n_aoa + phi_LOS_AOA
phi_nm_AOA = np.zeros((N_active, M_rays))
for n in range(N_active):
    phi_nm_AOA[n, :] = phi_n_AOA[n] + c_ASA * alpha_m
phi_AOD_prime = 2.0 * (ASD / 1.4) * np.sqrt(-np.log(P_n / np.max(P_n))) / C_phi_NLOS
X_n_aod = np.random.choice([-1, 1], N_active)
Y_n_aod = np.random.normal(0, ASD / 7.0, N_active)
phi_n_AOD = X_n_aod * phi_AOD_prime + Y_n_aod + phi_LOS_AOD
phi_nm_AOD = np.zeros((N_active, M_rays))
for n in range(N_active):
    phi_nm_AOD[n, :] = phi_n_AOD[n] + c_ASD * alpha_m
theta_ZOA_prime = -ZSA * np.log(P_n / np.max(P_n)) / C_theta_NLOS
X_n_zoa = np.random.choice([-1, 1], N_active)
Y_n_zoa = np.random.normal(0, ZSA / 7.0, N_active)
theta_n_ZOA = X_n_zoa * theta_ZOA_prime + Y_n_zoa + theta_LOS_ZOA
theta_nm_ZOA = np.zeros((N_active, M_rays))
for n in range(N_active):
    theta_nm_ZOA[n, :] = theta_n_ZOA[n] + c_ZSA * alpha_m
theta_nm_ZOA = theta_nm_ZOA % 360.0
mask = theta_nm_ZOA > 180.0
theta_nm_ZOA[mask] = 360.0 - theta_nm_ZOA[mask]
a_fc = 0.208 * np.log10(fc_lsp) - 0.782
b_fc = 25.0
c_fc = -0.13 * np.log10(fc_lsp) + 2.03
e_fc = 7.66 * np.log10(fc_lsp) - 5.96
mu_offset_ZOD = e_fc - 10**(a_fc * np.log10(max(b_fc, d_2D)) + c_fc - 0.07 * (h_UT - 1.5))
theta_ZOD_prime = -ZSD * np.log(P_n / np.max(P_n)) / C_theta_NLOS
X_n_zod = np.random.choice([-1, 1], N_active)
Y_n_zod = np.random.normal(0, ZSD / 7.0, N_active)
theta_n_ZOD = X_n_zod * theta_ZOD_prime + Y_n_zod + theta_LOS_ZOD + mu_offset_ZOD
theta_nm_ZOD = np.zeros((N_active, M_rays))
for n in range(N_active):
    theta_nm_ZOD[n, :] = theta_n_ZOD[n] + (3.0/8.0) * (10**mu_lgZSD) * alpha_m
print(f"\nSTEP 7: Arrival/Departure angles generated")
print(f"  AOA range: [{phi_nm_AOA.min():.1f}, {phi_nm_AOA.max():.1f}]°")
print(f"  AOD range: [{phi_nm_AOD.min():.1f}, {phi_nm_AOD.max():.1f}]°")
print(f"  ZOA range: [{theta_nm_ZOA.min():.1f}, {theta_nm_ZOA.max():.1f}]°")
print(f"  ZOD range: [{theta_nm_ZOD.min():.1f}, {theta_nm_ZOD.max():.1f}]°")
print(f"  mu_offset_ZOD = {mu_offset_ZOD:.2f}°")
for n in range(N_active):
    perm_aoa = np.random.permutation(M_rays)
    perm_zoa = np.random.permutation(M_rays)
    perm_zod = np.random.permutation(M_rays)
    phi_nm_AOA[n, :] = phi_nm_AOA[n, perm_aoa]
    theta_nm_ZOA[n, :] = theta_nm_ZOA[n, perm_zoa]
    theta_nm_ZOD[n, :] = theta_nm_ZOD[n, perm_zod]
print(f"\nSTEP 8: Ray coupling within clusters completed")
mu_XPR = 7.0
sigma_XPR = 3.0
X_nm = np.random.normal(mu_XPR, sigma_XPR, (N_active, M_rays))
kappa_nm = 10**(X_nm / 10.0)
print(f"\nSTEP 9: XPR generated")
print(f"  Mean XPR = {10*np.log10(np.mean(kappa_nm)):.1f} dB")
Phi_nm = {}
for pol in ['tt', 'tp', 'pt', 'pp']:
    Phi_nm[pol] = np.random.uniform(-np.pi, np.pi, (N_active, M_rays))
print(f"\nSTEP 10: Random initial phases drawn (4 pol combinations)")
def antenna_field_pattern(theta_deg, phi_deg, pol='V'):
    if pol == 'V':
        return 1.0, 0.0
    else:
        return 0.0, 1.0
v_speed = 3.0 / 3.6
phi_v = 0.0
theta_v = 90.0
v_bar = v_speed * np.array([
    np.sin(np.radians(theta_v)) * np.cos(np.radians(phi_v)),
    np.sin(np.radians(theta_v)) * np.sin(np.radians(phi_v)),
    np.cos(np.radians(theta_v))
])
t = 0.0
def spherical_unit_vector(theta_deg, phi_deg):
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    return np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
H = np.zeros((U, S), dtype=complex)
bs_pol = []
for col in range(N_BS_col):
    bs_pol.append('V')
    bs_pol.append('H')
ut_pol = ['V', 'V']
for n in range(N_active):
    sqrt_Pn = np.sqrt(P_n[n])
    for m in range(M_rays):
        r_rx = spherical_unit_vector(theta_nm_ZOA[n, m], phi_nm_AOA[n, m])
        r_tx = spherical_unit_vector(theta_nm_ZOD[n, m], phi_nm_AOD[n, m])
        kappa = kappa_nm[n, m]
        pol_matrix = np.array([
            [np.exp(1j * Phi_nm['tt'][n, m]),
             np.sqrt(1.0/kappa) * np.exp(1j * Phi_nm['tp'][n, m])],
            [np.sqrt(1.0/kappa) * np.exp(1j * Phi_nm['pt'][n, m]),
             np.exp(1j * Phi_nm['pp'][n, m])]
        ])
        v_doppler = np.exp(1j * 2 * np.pi * np.dot(r_rx, v_bar) * t / lambda_0)
        for u_idx in range(U):
            F_rx_theta, F_rx_phi = antenna_field_pattern(
                theta_nm_ZOA[n, m], phi_nm_AOA[n, m], ut_pol[u_idx])
            F_rx = np.array([F_rx_theta, F_rx_phi])
            rx_phase = np.exp(1j * 2 * np.pi * np.dot(r_rx, d_rx[u_idx]) / lambda_0)
            for s_idx in range(S):
                F_tx_theta, F_tx_phi = antenna_field_pattern(
                    theta_nm_ZOD[n, m], phi_nm_AOD[n, m], bs_pol[s_idx])
                F_tx = np.array([F_tx_theta, F_tx_phi])
                tx_phase = np.exp(1j * 2 * np.pi * np.dot(r_tx, d_tx[s_idx]) / lambda_0)
                H[u_idx, s_idx] += (sqrt_Pn / np.sqrt(M_rays)) * \
                    (F_rx @ pol_matrix @ F_tx) * rx_phase * tx_phase * v_doppler
print(f"\nSTEP 11: Channel matrix H ({U}x{S}) generated")
H_dB = 20 * np.log10(np.abs(H))
print(f"  |H| magnitude [dB]:")
print(np.round(H_dB, 1))
_, sv, _ = np.linalg.svd(H)
cond_number = sv[0] / sv[-1] if sv[-1] > 0 else float('inf')
print(f"  Condition number: {cond_number:.2f} ({20*np.log10(cond_number):.1f} dB)")
PL_total = PL_uma_nlos + SF
PL_linear = 10**(-PL_total / 20.0)
H_final = H * PL_linear
print(f"\nSTEP 12: Path loss applied")
print(f"  PL + SF = {PL_total:.2f} dB")
print(f"  |H_final| mean = {20*np.log10(np.mean(np.abs(H_final))):.1f} dB")
rms_ds = np.sqrt(np.sum(P_n * tau**2) - (np.sum(P_n * tau))**2)
print(f"\nRESULTS:")
print(f"  RMS Delay Spread: {rms_ds*1e9:.2f} ns")
print(f"  Number of active clusters: {N_active}")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax1 = axes[0]
markerline, stemlines, baseline = ax1.stem(tau * 1e9, 10 * np.log10(P_n), basefmt=" ")
stemlines.set_linewidth(1.5)
markerline.set_markersize(5)
ax1.set_xlabel('Delay [ns]')
ax1.set_ylabel('Normalized Power [dB]')
ax1.set_title(f'PDP — UMa NLOS (RMS DS = {rms_ds*1e9:.0f} ns)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=-30)
ax2 = axes[1]
for n in range(N_active):
    for m in range(M_rays):
        ax2.scatter(phi_nm_AOA[n, m], 10*np.log10(P_n[n]/M_rays),
                   c='steelblue', s=8, alpha=0.5)
ax2.set_xlabel('AOA [deg]')
ax2.set_ylabel('Power [dB]')
ax2.set_title('Angular Power Spectrum (Azimuth)')
ax2.grid(True, alpha=0.3)
ax3 = axes[2]
im = ax3.imshow(H_dB, aspect='auto', cmap='viridis')
fig.colorbar(im, ax=ax3, label='Magnitude [dB]')
ax3.set_xlabel('BS Antenna Index (4×2 Cross-Pol)')
ax3.set_ylabel('UT Antenna Index')
ax3.set_title(f'Channel Matrix |H| ({U}×{S} MIMO)')
plt.tight_layout()
plt.savefig('public/pdp_plot.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: public/pdp_plot.png (PDP + APS + Heatmap)")
fig2, ax = plt.subplots(figsize=(8, 3))
im2 = ax.imshow(H_dB, aspect='auto', cmap='viridis')
fig2.colorbar(im2, ax=ax, label='Magnitude [dB]')
ax.set_xlabel('BS Antenna Index (4×2 Cross-Pol)')
ax.set_ylabel('UT Antenna Index')
ax.set_title(f'Channel Matrix |H| Heatmap (Step 11)')
for i in range(U):
    for j in range(S):
        ax.text(j, i, f'{H_dB[i,j]:.1f}', ha='center', va='center',
                fontsize=7, color='white' if H_dB[i,j] < np.mean(H_dB) else 'black')
plt.tight_layout()
plt.savefig('public/h_matrix_heatmap.png', dpi=150, bbox_inches='tight')
print("Plot saved: public/h_matrix_heatmap.png (Heatmap)")
plt.show()
if __name__ == "__main__":
    pass
