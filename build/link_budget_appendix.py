import math
import numpy as np
f_c_ghz = 1.8
f_c_hz  = 1.8e9
c       = 3.0e8
P_t_dBm       = 46.0
G_ant_dBi      = 18.0
L_cable_dB     = 2.0
P_r_min_dBm    = -104.0
L_sf_dB        = 8.0
L_penetration  = 0.0
h_BS  = 25.0
h_UT  = 1.5
h_building = 8.0
W_street   = 15.0
town_area_km2 = 8.0
def compute_lmax():
    L_max_simple = P_t_dBm - P_r_min_dBm
    L_max_extended = (P_t_dBm + G_ant_dBi - L_cable_dB
                      - P_r_min_dBm - L_sf_dB - L_penetration)
    return L_max_simple, L_max_extended
def uma_breakpoint_distance():
    h_E = 1.0
    h_BS_eff = h_BS - h_E
    h_UT_eff = h_UT - h_E
    d_BP = 2 * math.pi * h_BS_eff * h_UT_eff * f_c_hz / c
    return d_BP, h_BS_eff, h_UT_eff
def pl_uma_los(d_3D):
    d_BP, _, _ = uma_breakpoint_distance()
    delta_h = h_BS - h_UT
    d_2D = math.sqrt(max(d_3D**2 - delta_h**2, 0))
    if d_2D <= d_BP:
        PL = 28.0 + 22 * math.log10(d_3D) + 20 * math.log10(f_c_ghz)
    else:
        PL = (28.0 + 40 * math.log10(d_3D) + 20 * math.log10(f_c_ghz)
              - 9 * math.log10(d_BP**2 + delta_h**2))
    return PL
def pl_uma_nlos(d_3D):
    PL_nlos_prime = (13.54
                     + 39.08 * math.log10(d_3D)
                     + 20 * math.log10(f_c_ghz)
                     - 0.6 * (h_UT - 1.5))
    PL_los = pl_uma_los(d_3D)
    return max(PL_los, PL_nlos_prime)
def pl_umi_los(d_3D, h_bs_umi=10.0):
    h_E = 1.0
    h_bs_eff = h_bs_umi - h_E
    h_ut_eff = h_UT - h_E
    d_BP = 2 * math.pi * h_bs_eff * h_ut_eff * f_c_hz / c
    delta_h = h_bs_umi - h_UT
    d_2D = math.sqrt(max(d_3D**2 - delta_h**2, 0))
    if d_2D <= d_BP:
        PL = 32.4 + 21 * math.log10(d_3D) + 20 * math.log10(f_c_ghz)
    else:
        PL = (32.4 + 40 * math.log10(d_3D) + 20 * math.log10(f_c_ghz)
              - 9.5 * math.log10(d_BP**2 + delta_h**2))
    return PL
def pl_umi_nlos(d_3D, h_bs_umi=10.0):
    PL_nlos_prime = (22.4
                     + 35.3 * math.log10(d_3D)
                     + 21.3 * math.log10(f_c_ghz)
                     - 0.3 * (h_UT - 1.5))
    PL_los = pl_umi_los(d_3D, h_bs_umi)
    return max(PL_los, PL_nlos_prime)
def pl_rma_los(d_3D, h_bs_rma=35.0):
    h = h_building
    d_BP_rma = 2 * math.pi * h_bs_rma * h_UT * f_c_hz / c
    delta_h = h_bs_rma - h_UT
    d_2D = math.sqrt(max(d_3D**2 - delta_h**2, 0))
    term1 = 20 * math.log10(40 * math.pi * d_3D * f_c_ghz / 3)
    term2 = min(0.03 * h**1.72, 10) * math.log10(d_3D)
    term3 = min(0.044 * h**1.72, 14.77)
    term4 = 0.002 * math.log10(h) * d_3D
    PL1 = term1 + term2 - term3 + term4
    if d_2D <= d_BP_rma:
        return PL1
    else:
        d_3D_bp = math.sqrt(d_BP_rma**2 + delta_h**2)
        term1_bp = 20 * math.log10(40 * math.pi * d_3D_bp * f_c_ghz / 3)
        term2_bp = min(0.03 * h**1.72, 10) * math.log10(d_3D_bp)
        term4_bp = 0.002 * math.log10(h) * d_3D_bp
        PL1_bp = term1_bp + term2_bp - term3 + term4_bp
        PL2 = PL1_bp + 40 * math.log10(d_3D / d_3D_bp)
        return PL2
def pl_rma_nlos(d_3D, h_bs_rma=35.0):
    W = W_street
    h = h_building
    PL_prime = (161.04
                - 7.1 * math.log10(W)
                + 7.5 * math.log10(h)
                - (24.37 - 3.7 * (h / h_bs_rma)**2) * math.log10(h_bs_rma)
                + (43.42 - 3.1 * math.log10(h_bs_rma)) * (math.log10(d_3D) - 3)
                + 20 * math.log10(f_c_ghz)
                - (3.2 * (math.log10(11.75 * h_UT))**2 - 4.97))
    PL_los = pl_rma_los(d_3D, h_bs_rma)
    return max(PL_los, PL_prime)
def solve_rmax_uma_nlos(L_max):
    log10_fc = math.log10(f_c_ghz)
    ut_offset = 0.6 * (h_UT - 1.5)
    log10_d = (L_max - 13.54 - 20 * log10_fc + ut_offset) / 39.08
    d_max = 10**log10_d
    return d_max
def solve_rmax_umi_nlos(L_max):
    log10_fc = math.log10(f_c_ghz)
    ut_offset = 0.3 * (h_UT - 1.5)
    log10_d = (L_max - 22.4 - 21.3 * log10_fc + ut_offset) / 35.3
    d_max = 10**log10_d
    return d_max
def solve_rmax_numerical(pl_func, L_max, d_min=10, d_max=10000):
    for _ in range(100):
        d_mid = (d_min + d_max) / 2
        pl = pl_func(d_mid)
        if abs(pl - L_max) < 0.01:
            return d_mid
        elif pl < L_max:
            d_min = d_mid
        else:
            d_max = d_mid
    return (d_min + d_max) / 2
def hex_cell_area(R):
    return (3 * math.sqrt(3) / 2) * R**2
def cells_required(total_area_km2, R_km, overlap_factor=1.15):
    cell_area = hex_cell_area(R_km)
    n_cells = math.ceil(total_area_km2 * overlap_factor / cell_area)
    return n_cells, cell_area
def main():
    print("=" * 72)
    print("   4G LTE Radio Network Planning — Kagal, Kolhapur")
    print("   Reference: 3GPP TR 38.901 V18.0.0, Table 7.4.1-1")
    print("=" * 72)
    L_max_simple, L_max_extended = compute_lmax()
    print("\n1. LINK BUDGET")
    print("-" * 40)
    print(f"   Transmit Power (P_t)       = {P_t_dBm:>8.1f} dBm")
    print(f"   Antenna Gain (G_ant)       = {G_ant_dBi:>8.1f} dBi")
    print(f"   Cable Loss (L_cable)       = {L_cable_dB:>8.1f} dB")
    print(f"   Rx Sensitivity (P_r,min)   = {P_r_min_dBm:>8.1f} dBm")
    print(f"   Shadow Fading Margin       = {L_sf_dB:>8.1f} dB")
    print(f"   Carrier Frequency (f_c)    = {f_c_ghz:>8.1f} GHz")
    print()
    print(f"   L_max (simple: Pt - Pr)    = {L_max_simple:>8.1f} dB")
    print(f"   L_max (extended)           = {L_max_extended:>8.1f} dB")
    print(f"   → Using L_max = {L_max_extended:.1f} dB for planning")
    L_max = L_max_extended
    d_BP, h_BS_eff, h_UT_eff = uma_breakpoint_distance()
    print(f"\n2. UMa BREAKPOINT DISTANCE")
    print("-" * 40)
    print(f"   h_E (effective env height) =    1.0 m")
    print(f"   h'_BS = h_BS - h_E         = {h_BS_eff:>6.1f} m")
    print(f"   h'_UT = h_UT - h_E         = {h_UT_eff:>6.1f} m")
    print(f"   d'_BP = 2π·h'_BS·h'_UT·f_c/c")
    print(f"         = 2π × {h_BS_eff:.1f} × {h_UT_eff:.1f} × {f_c_hz:.1e} / {c:.1e}")
    print(f"         = {d_BP:.1f} m")
    R_max_uma = solve_rmax_uma_nlos(L_max)
    print(f"\n3. MAXIMUM CELL RADIUS — UMa NLOS")
    print("-" * 40)
    print(f"   Formula: PL = 13.54 + 39.08·log10(d) + 20·log10(f_c)")
    print(f"                 - 0.6·(h_UT - 1.5)")
    print(f"   Setting PL = L_max = {L_max:.1f} dB:")
    log10_fc = math.log10(f_c_ghz)
    log10_d = (L_max - 13.54 - 20 * log10_fc) / 39.08
    print(f"   log10(d) = ({L_max:.1f} - 13.54 - 20×{log10_fc:.4f}) / 39.08")
    print(f"            = {log10_d:.4f}")
    print(f"   d = 10^{log10_d:.4f}")
    print(f"   R_max (UMa NLOS) = {R_max_uma:.1f} m = {R_max_uma/1000:.3f} km")
    pl_at_rmax = pl_uma_nlos(R_max_uma)
    print(f"\n   Verification: PL_UMa_NLOS({R_max_uma:.0f} m) = {pl_at_rmax:.2f} dB")
    print(f"   L_max = {L_max:.1f} dB  →  {'✓ PASS' if abs(pl_at_rmax - L_max) < 0.5 else '✗ FAIL'}")
    R_max_umi = solve_rmax_umi_nlos(L_max)
    print(f"\n4. MAXIMUM CELL RADIUS — UMi NLOS (microcell reference)")
    print("-" * 40)
    print(f"   Formula: PL = 22.4 + 35.3·log10(d) + 21.3·log10(f_c)")
    print(f"                 - 0.3·(h_UT - 1.5)")
    print(f"   R_max (UMi NLOS) = {R_max_umi:.1f} m = {R_max_umi/1000:.3f} km")
    R_max_rma = solve_rmax_numerical(pl_rma_nlos, L_max, d_min=100, d_max=10000)
    print(f"\n5. MAXIMUM CELL RADIUS — RMa NLOS (rural reference)")
    print("-" * 40)
    print(f"   R_max (RMa NLOS) = {R_max_rma:.1f} m = {R_max_rma/1000:.3f} km")
    print(f"   (Solved numerically via bisection)")
    R_km = R_max_uma / 1000
    n_cells, cell_area = cells_required(town_area_km2, R_km)
    print(f"\n6. CELL COUNT — Hexagonal Layout")
    print("-" * 40)
    print(f"   Town area                  = {town_area_km2:.1f} km²")
    print(f"   R_max                      = {R_max_uma:.0f} m = {R_km:.3f} km")
    print(f"   Hex cell area = 2.598·R²   = {cell_area:.4f} km²")
    print(f"   With 15% overlap margin:")
    print(f"   Cells required             = ⌈{town_area_km2:.1f} × 1.15 / {cell_area:.4f}⌉ = {n_cells}")
    for N in [1, 3, 4, 7]:
        D_over_R = math.sqrt(3 * N)
        D = D_over_R * R_max_uma
        print(f"\n   Reuse N={N}: D/R = √(3×{N}) = {D_over_R:.3f}, "
              f"D = {D:.0f} m, cells for {town_area_km2} km² = {n_cells}")
    print(f"\n7. PATH LOSS vs DISTANCE (UMa NLOS)")
    print("-" * 50)
    print(f"   {'Distance [m]':>14}  {'PL [dB]':>10}  {'Status':>10}")
    print(f"   {'─'*14}  {'─'*10}  {'─'*10}")
    test_distances = [100, 200, 300, 500, 750, 1000, 1200, 1500, 2000, 3000, 5000]
    for d in test_distances:
        pl = pl_uma_nlos(d)
        status = "< L_max ✓" if pl < L_max else "> L_max ✗"
        print(f"   {d:>14.0f}  {pl:>10.2f}  {status:>10}")
    print(f"\n8. PER-CELL VERIFICATION (Proposed Sites)")
    print("-" * 60)
    cells = [
        ("Site 1: Town Center/Market",     700),
        ("Site 2: NH-48 North Junction",    850),
        ("Site 3: NH-48 South / Bus Stand", 800),
        ("Site 4: East Residential",        750),
        ("Site 5: Dudhganga River Area",    900),
    ]
    print(f"   {'Cell':>35}  {'d_max [m]':>10}  {'PL [dB]':>8}  {'R_max':>8}  {'Check':>8}")
    print(f"   {'─'*35}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
    all_pass = True
    for name, d_farthest in cells:
        pl = pl_uma_nlos(d_farthest)
        status = "PASS ✓" if d_farthest < R_max_uma else "FAIL ✗"
        if d_farthest >= R_max_uma:
            all_pass = False
        print(f"   {name:>35}  {d_farthest:>10.0f}  {pl:>8.2f}  "
              f"{R_max_uma:>8.0f}  {status:>8}")
    print()
    if all_pass:
        print("   ★ All cells satisfy R < R_max. Network design is valid.")
    else:
        print("   ✗ Some cells exceed R_max. Redesign required!")
    print(f"\n{'=' * 72}")
    print(f"   SUMMARY")
    print(f"{'=' * 72}")
    print(f"   L_max (extended link budget) = {L_max:.1f} dB")
    print(f"   R_max (UMa NLOS)             = {R_max_uma:.0f} m ({R_max_uma/1000:.3f} km)")
    print(f"   R_max (UMi NLOS, microcell)  = {R_max_umi:.0f} m ({R_max_umi/1000:.3f} km)")
    print(f"   R_max (RMa NLOS, rural)      = {R_max_rma:.0f} m ({R_max_rma/1000:.3f} km)")
    print(f"   Town area                    = {town_area_km2:.1f} km²")
    print(f"   Hex cell area                = {cell_area:.4f} km²")
    print(f"   Cells required               = {n_cells}")
    print(f"   All cells R < R_max?         = {'YES ✓' if all_pass else 'NO ✗'}")
    print(f"{'=' * 72}")
if __name__ == "__main__":
    main()
