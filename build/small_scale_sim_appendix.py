import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fc = 1.8e9
fc_ghz = 1.8
c = 3e8
lam = c / fc

Ncol, Npol = 4, 2
S = Ncol * Npol
U = 2
dH = 0.5 * lam
dV = 0.5 * lam

hBS, hUT = 25.0, 1.5
bs = np.array([0.0, 0.0, hBS])
ut = np.array([500.0, 300.0, hUT])

d2D = np.linalg.norm(ut[:2] - bs[:2])
d3D = np.linalg.norm(ut - bs)
delta = ut - bs
aod_los = np.degrees(np.arctan2(delta[1], delta[0]))
aoa_los = aod_los + 180.0
zod_los = np.degrees(np.arccos(delta[2] / d3D))
zoa_los = 180.0 - zod_los

dtx = np.zeros((S, 3))
for col in range(Ncol):
    dtx[col*2] = [col * dH, 0, 0]
    dtx[col*2 + 1] = [col * dH, 0, dV]

drx = np.zeros((U, 3))
drx[1] = [dH, 0, 0]

print("STEP 1: Setup")
print(f"  UMa NLOS | fc={fc_ghz} GHz | lambda={lam*100:.2f} cm")
print(f"  BS: {S} elements | UT: {U} elements")
print(f"  d_2D={d2D:.1f} m | d_3D={d3D:.1f} m")
print(f"  AOD={aod_los:.1f} AOA={aoa_los:.1f} ZOD={zod_los:.1f} ZOA={zoa_los:.1f}")

prop = "NLOS"
print(f"\nSTEP 2: Propagation = {prop}")

PL = 13.54 + 39.08*np.log10(d3D) + 20*np.log10(fc_ghz) - 0.6*(hUT - 1.5)
sf_sigma = 6.0
SF = np.random.normal(0, sf_sigma)
print(f"\nSTEP 3: PL = {PL:.2f} dB | SF = {SF:.2f} dB")

fc_lsp = max(fc_ghz, 6.0)

mu_ds  = -6.28 - 0.204*np.log10(fc_lsp)
mu_asd = 1.5 - 0.1144*np.log10(fc_lsp)
mu_asa = 2.08 - 0.27*np.log10(fc_lsp)
mu_zsa = -0.3236*np.log10(fc_lsp) + 1.512
mu_zsd = max(-0.5, -2.1*(d2D/1000) - 0.01*(hUT - 1.5) + 0.9)

sig_ds, sig_asd, sig_asa = 0.39, 0.28, 0.11
sig_zsa, sig_zsd = 0.16, 0.49

corr = np.array([
    [ 1.0,  0.4,  0.6,  0.0, -0.5],
    [ 0.4,  1.0,  0.4, -0.1,  0.5],
    [ 0.6,  0.4,  1.0,  0.0,  0.0],
    [ 0.0, -0.1,  0.0,  1.0,  0.0],
    [-0.5,  0.5,  0.0,  0.0,  1.0],
])
L = np.linalg.cholesky(corr)
z = np.random.randn(5)
s = L @ z

DS  = 10**(mu_ds  + sig_ds  * s[0])
ASD = min(10**(mu_asd + sig_asd * s[1]), 104.0)
ASA = min(10**(mu_asa + sig_asa * s[2]), 104.0)
ZSA = min(10**(mu_zsa + sig_zsa * s[3]), 52.0)
ZSD = min(10**(mu_zsd + sig_zsd * s[4]), 52.0)

print(f"\nSTEP 4: LSPs")
print(f"  DS={DS*1e9:.2f} ns | ASD={ASD:.2f} | ASA={ASA:.2f}")
print(f"  ZSA={ZSA:.2f} | ZSD={ZSD:.2f}")

Nclust = 20
Mrays = 20
rtau = 2.3

xn = np.random.uniform(0, 1, Nclust)
tau_raw = -rtau * DS * np.log(xn)
tau = np.sort(tau_raw - tau_raw.min())
print(f"\nSTEP 5: Delays")
print(f"  min={tau[0]*1e9:.2f} ns | max={tau[-1]*1e9:.2f} ns")

zeta = 3.0
Zn = np.random.normal(0, zeta, Nclust)
Pprime = np.exp(-tau*(rtau - 1)/(rtau*DS)) * 10**(-Zn/10)
Pn = Pprime / Pprime.sum()
threshold = Pn.max() * 10**(-2.5)
keep = Pn >= threshold
tau, Pn = tau[keep], Pn[keep]
Pn /= Pn.sum()
Nact = len(tau)
print(f"\nSTEP 6: Powers")
print(f"  Active clusters: {Nact}/{Nclust}")
print(f"  Strongest: {10*np.log10(Pn[0]):.1f} dB")

offsets = np.array([
    0.0447, -0.0447, 0.1413, -0.1413,
    0.2492, -0.2492, 0.3715, -0.3715,
    0.5129, -0.5129, 0.6797, -0.6797,
    0.8844, -0.8844, 1.1481, -1.1481,
    1.5195, -1.5195, 2.1551, -2.1551
])

cASD, cASA, cZSA = 2.0, 15.0, 7.0
Cphi, Ctheta = 1.289, 1.178

aoa_cluster = 2*(ASA/1.4)*np.sqrt(-np.log(Pn/Pn.max())) / Cphi
Xaoa = np.random.choice([-1, 1], Nact)
Yaoa = np.random.normal(0, ASA/7, Nact)
phi_aoa = Xaoa*aoa_cluster + Yaoa + aoa_los

phi_aoa_rays = np.zeros((Nact, Mrays))
for n in range(Nact):
    phi_aoa_rays[n] = phi_aoa[n] + cASA * offsets

aod_cluster = 2*(ASD/1.4)*np.sqrt(-np.log(Pn/Pn.max())) / Cphi
Xaod = np.random.choice([-1, 1], Nact)
Yaod = np.random.normal(0, ASD/7, Nact)
phi_aod = Xaod*aod_cluster + Yaod + aod_los

phi_aod_rays = np.zeros((Nact, Mrays))
for n in range(Nact):
    phi_aod_rays[n] = phi_aod[n] + cASD * offsets

zoa_cluster = -ZSA * np.log(Pn/Pn.max()) / Ctheta
Xzoa = np.random.choice([-1, 1], Nact)
Yzoa = np.random.normal(0, ZSA/7, Nact)
th_zoa = Xzoa*zoa_cluster + Yzoa + zoa_los

th_zoa_rays = np.zeros((Nact, Mrays))
for n in range(Nact):
    th_zoa_rays[n] = th_zoa[n] + cZSA * offsets
th_zoa_rays = th_zoa_rays % 360
th_zoa_rays[th_zoa_rays > 180] = 360 - th_zoa_rays[th_zoa_rays > 180]

a_f = 0.208*np.log10(fc_lsp) - 0.782
b_f = 25.0
c_f = -0.13*np.log10(fc_lsp) + 2.03
e_f = 7.66*np.log10(fc_lsp) - 5.96
mu_zod_off = e_f - 10**(a_f*np.log10(max(b_f, d2D)) + c_f - 0.07*(hUT - 1.5))

zod_cluster = -ZSD * np.log(Pn/Pn.max()) / Ctheta
Xzod = np.random.choice([-1, 1], Nact)
Yzod = np.random.normal(0, ZSD/7, Nact)
th_zod = Xzod*zod_cluster + Yzod + zod_los + mu_zod_off

th_zod_rays = np.zeros((Nact, Mrays))
for n in range(Nact):
    th_zod_rays[n] = th_zod[n] + (3/8)*(10**mu_zsd)*offsets

print(f"\nSTEP 7: Angles")
print(f"  AOA: [{phi_aoa_rays.min():.1f}, {phi_aoa_rays.max():.1f}]")
print(f"  AOD: [{phi_aod_rays.min():.1f}, {phi_aod_rays.max():.1f}]")
print(f"  ZOA: [{th_zoa_rays.min():.1f}, {th_zoa_rays.max():.1f}]")
print(f"  ZOD: [{th_zod_rays.min():.1f}, {th_zod_rays.max():.1f}]")

for n in range(Nact):
    p1 = np.random.permutation(Mrays)
    p2 = np.random.permutation(Mrays)
    p3 = np.random.permutation(Mrays)
    phi_aoa_rays[n] = phi_aoa_rays[n, p1]
    th_zoa_rays[n] = th_zoa_rays[n, p2]
    th_zod_rays[n] = th_zod_rays[n, p3]
print(f"\nSTEP 8: Ray coupling done")

mu_xpr, sig_xpr = 7.0, 3.0
Xnm = np.random.normal(mu_xpr, sig_xpr, (Nact, Mrays))
kappa = 10**(Xnm / 10)
print(f"\nSTEP 9: XPR mean = {10*np.log10(kappa.mean()):.1f} dB")

phases = {}
for key in ['tt', 'tp', 'pt', 'pp']:
    phases[key] = np.random.uniform(-np.pi, np.pi, (Nact, Mrays))
print(f"STEP 10: Initial phases drawn")


def field(theta, phi, pol):
    if pol == 'V':
        return 1.0, 0.0
    return 0.0, 1.0


def rhat(th, ph):
    t, p = np.radians(th), np.radians(ph)
    return np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])


v = (3.0/3.6) * np.array([1, 0, 0])
t = 0.0

H = np.zeros((U, S), dtype=complex)

bpol = []
for col in range(Ncol):
    bpol += ['V', 'H']
upol = ['V', 'V']

for n in range(Nact):
    sqP = np.sqrt(Pn[n])
    for m in range(Mrays):
        rrx = rhat(th_zoa_rays[n, m], phi_aoa_rays[n, m])
        rtx = rhat(th_zod_rays[n, m], phi_aod_rays[n, m])
        k = kappa[n, m]
        M = np.array([
            [np.exp(1j*phases['tt'][n,m]),
             np.sqrt(1/k)*np.exp(1j*phases['tp'][n,m])],
            [np.sqrt(1/k)*np.exp(1j*phases['pt'][n,m]),
             np.exp(1j*phases['pp'][n,m])]
        ])
        dopp = np.exp(1j * 2*np.pi * np.dot(rrx, v) * t / lam)
        for u in range(U):
            Frt, Frp = field(th_zoa_rays[n,m], phi_aoa_rays[n,m], upol[u])
            Fr = np.array([Frt, Frp])
            rx_ph = np.exp(1j*2*np.pi*np.dot(rrx, drx[u]) / lam)
            for si in range(S):
                Ftt, Ftp = field(th_zod_rays[n,m], phi_aod_rays[n,m], bpol[si])
                Ft = np.array([Ftt, Ftp])
                tx_ph = np.exp(1j*2*np.pi*np.dot(rtx, dtx[si]) / lam)
                H[u, si] += (sqP/np.sqrt(Mrays)) * (Fr @ M @ Ft) * rx_ph * tx_ph * dopp

print(f"\nSTEP 11: H matrix ({U}x{S})")
HdB = 20 * np.log10(np.abs(H))
print(np.round(HdB, 1))
_, sv, _ = np.linalg.svd(H)
cn = sv[0]/sv[-1] if sv[-1] > 0 else float('inf')
print(f"  Condition number: {cn:.2f} ({20*np.log10(cn):.1f} dB)")

PL_total = PL + SF
Hfinal = H * 10**(-PL_total/20)
print(f"\nSTEP 12: Path loss applied")
print(f"  Total PL+SF = {PL_total:.2f} dB")
print(f"  |H_final| mean = {20*np.log10(np.abs(Hfinal).mean()):.1f} dB")

rms_ds = np.sqrt(np.sum(Pn * tau**2) - np.sum(Pn*tau)**2)
print(f"\nResults:")
print(f"  RMS delay spread = {rms_ds*1e9:.2f} ns")
print(f"  Active clusters  = {Nact}")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ml, sl, bl = ax[0].stem(tau*1e9, 10*np.log10(Pn), basefmt=" ")
sl.set_linewidth(1.5)
ml.set_markersize(5)
ax[0].set_xlabel('Delay [ns]')
ax[0].set_ylabel('Power [dB]')
ax[0].set_title(f'PDP (RMS DS = {rms_ds*1e9:.0f} ns)')
ax[0].grid(True, alpha=0.3)
ax[0].set_ylim(bottom=-30)

for n in range(Nact):
    for m in range(Mrays):
        ax[1].scatter(phi_aoa_rays[n,m], 10*np.log10(Pn[n]/Mrays),
                      c='steelblue', s=8, alpha=0.5)
ax[1].set_xlabel('AOA [deg]')
ax[1].set_ylabel('Power [dB]')
ax[1].set_title('Angular Power Spectrum')
ax[1].grid(True, alpha=0.3)

im = ax[2].imshow(HdB, aspect='auto', cmap='viridis')
fig.colorbar(im, ax=ax[2], label='|H| [dB]')
ax[2].set_xlabel('BS Antenna')
ax[2].set_ylabel('UT Antenna')
ax[2].set_title(f'|H| ({U}x{S} MIMO)')
plt.tight_layout()
plt.savefig('public/pdp_plot.png', dpi=150, bbox_inches='tight')
print("\nSaved: public/pdp_plot.png")

fig2, ax2 = plt.subplots(figsize=(8, 3))
im2 = ax2.imshow(HdB, aspect='auto', cmap='viridis')
fig2.colorbar(im2, ax=ax2, label='|H| [dB]')
ax2.set_xlabel('BS Antenna')
ax2.set_ylabel('UT Antenna')
ax2.set_title('H Matrix Heatmap')
for i in range(U):
    for j in range(S):
        ax2.text(j, i, f'{HdB[i,j]:.1f}', ha='center', va='center',
                 fontsize=7, color='white' if HdB[i,j] < HdB.mean() else 'black')
plt.tight_layout()
plt.savefig('public/h_matrix_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: public/h_matrix_heatmap.png")
