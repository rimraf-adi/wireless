import math
import numpy as np

fc = 1.8
fc_hz = 1.8e9
c = 3e8

Pt = 46.0
Gant = 18.0
Lcable = 2.0
Prmin = -104.0
Lsf = 8.0
Lpen = 0.0

hBS = 25.0
hUT = 1.5
hbldg = 8.0
Wst = 15.0
area = 8.0


def link_budget():
    simple = Pt - Prmin
    extended = Pt + Gant - Lcable - Prmin - Lsf - Lpen
    return simple, extended


def breakpoint_uma():
    hE = 1.0
    hbs = hBS - hE
    hut = hUT - hE
    dbp = 2 * math.pi * hbs * hut * fc_hz / c
    return dbp, hbs, hut


def pathloss_uma_los(d):
    dbp, _, _ = breakpoint_uma()
    dh = hBS - hUT
    d2d = math.sqrt(max(d**2 - dh**2, 0))
    if d2d <= dbp:
        return 28.0 + 22 * math.log10(d) + 20 * math.log10(fc)
    return (28.0 + 40 * math.log10(d) + 20 * math.log10(fc)
            - 9 * math.log10(dbp**2 + dh**2))


def pathloss_uma_nlos(d):
    pl_nlos = (13.54 + 39.08 * math.log10(d)
               + 20 * math.log10(fc) - 0.6 * (hUT - 1.5))
    pl_los = pathloss_uma_los(d)
    return max(pl_los, pl_nlos)


def pathloss_umi_los(d, hbs=10.0):
    hE = 1.0
    hb = hbs - hE
    hu = hUT - hE
    dbp = 2 * math.pi * hb * hu * fc_hz / c
    dh = hbs - hUT
    d2d = math.sqrt(max(d**2 - dh**2, 0))
    if d2d <= dbp:
        return 32.4 + 21 * math.log10(d) + 20 * math.log10(fc)
    return (32.4 + 40 * math.log10(d) + 20 * math.log10(fc)
            - 9.5 * math.log10(dbp**2 + dh**2))


def pathloss_umi_nlos(d, hbs=10.0):
    pl_nlos = (22.4 + 35.3 * math.log10(d)
               + 21.3 * math.log10(fc) - 0.3 * (hUT - 1.5))
    return max(pathloss_umi_los(d, hbs), pl_nlos)


def pathloss_rma_los(d, hbs=35.0):
    h = hbldg
    dbp = 2 * math.pi * hbs * hUT * fc_hz / c
    dh = hbs - hUT
    d2d = math.sqrt(max(d**2 - dh**2, 0))

    t1 = 20 * math.log10(40 * math.pi * d * fc / 3)
    t2 = min(0.03 * h**1.72, 10) * math.log10(d)
    t3 = min(0.044 * h**1.72, 14.77)
    t4 = 0.002 * math.log10(h) * d
    pl1 = t1 + t2 - t3 + t4

    if d2d <= dbp:
        return pl1

    d3bp = math.sqrt(dbp**2 + dh**2)
    t1b = 20 * math.log10(40 * math.pi * d3bp * fc / 3)
    t2b = min(0.03 * h**1.72, 10) * math.log10(d3bp)
    t4b = 0.002 * math.log10(h) * d3bp
    pl1b = t1b + t2b - t3 + t4b
    return pl1b + 40 * math.log10(d / d3bp)


def pathloss_rma_nlos(d, hbs=35.0):
    h = hbldg
    pl = (161.04 - 7.1 * math.log10(Wst) + 7.5 * math.log10(h)
          - (24.37 - 3.7 * (h / hbs)**2) * math.log10(hbs)
          + (43.42 - 3.1 * math.log10(hbs)) * (math.log10(d) - 3)
          + 20 * math.log10(fc)
          - (3.2 * (math.log10(11.75 * hUT))**2 - 4.97))
    return max(pathloss_rma_los(d, hbs), pl)


def rmax_uma(Lmax):
    lg_fc = math.log10(fc)
    offset = 0.6 * (hUT - 1.5)
    lg_d = (Lmax - 13.54 - 20 * lg_fc + offset) / 39.08
    return 10**lg_d


def rmax_umi(Lmax):
    lg_fc = math.log10(fc)
    offset = 0.3 * (hUT - 1.5)
    lg_d = (Lmax - 22.4 - 21.3 * lg_fc + offset) / 35.3
    return 10**lg_d


def rmax_bisection(plfn, Lmax, lo=10, hi=10000):
    for _ in range(100):
        mid = (lo + hi) / 2
        if abs(plfn(mid) - Lmax) < 0.01:
            return mid
        if plfn(mid) < Lmax:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def hex_area(R):
    return (3 * math.sqrt(3) / 2) * R**2


def num_cells(total_km2, R_km, overlap=1.15):
    ca = hex_area(R_km)
    return math.ceil(total_km2 * overlap / ca), ca


def main():
    print("=" * 60)
    print("  4G LTE Link Budget Calculator -- Kagal, Kolhapur")
    print("  3GPP TR 38.901 V18.0.0, Table 7.4.1-1")
    print("=" * 60)

    Lsimple, Lext = link_budget()

    print("\n--- Link Budget ---")
    print(f"  Pt        = {Pt:.1f} dBm")
    print(f"  G_ant     = {Gant:.1f} dBi")
    print(f"  L_cable   = {Lcable:.1f} dB")
    print(f"  Pr_min    = {Prmin:.1f} dBm")
    print(f"  L_sf      = {Lsf:.1f} dB")
    print(f"  f_c       = {fc:.1f} GHz")
    print(f"  L_max (simple)   = {Lsimple:.1f} dB")
    print(f"  L_max (extended) = {Lext:.1f} dB")

    Lmax = Lext
    dbp, hbs_eff, hut_eff = breakpoint_uma()

    print(f"\n--- UMa Breakpoint Distance ---")
    print(f"  h'_BS = {hbs_eff:.1f} m,  h'_UT = {hut_eff:.1f} m")
    print(f"  d'_BP = {dbp:.1f} m")

    R_uma = rmax_uma(Lmax)
    R_umi = rmax_umi(Lmax)
    R_rma = rmax_bisection(pathloss_rma_nlos, Lmax, lo=100, hi=10000)

    print(f"\n--- Maximum Cell Radius ---")
    print(f"  UMa NLOS:  R_max = {R_uma:.0f} m ({R_uma/1000:.2f} km)")
    print(f"  UMi NLOS:  R_max = {R_umi:.0f} m ({R_umi/1000:.2f} km)")
    print(f"  RMa NLOS:  R_max = {R_rma:.0f} m ({R_rma/1000:.2f} km)")

    pl_check = pathloss_uma_nlos(R_uma)
    print(f"\n  Verify: PL at R_max = {pl_check:.2f} dB (should equal {Lmax:.1f})")

    n, ca = num_cells(area, R_uma / 1000)
    print(f"\n--- Hex Layout ---")
    print(f"  Town area = {area:.1f} km2")
    print(f"  Cell area = {ca:.4f} km2")
    print(f"  Min cells = {n}")

    for N in [1, 3, 4, 7]:
        dr = math.sqrt(3 * N)
        print(f"  N={N}:  D/R = {dr:.3f},  D = {dr * R_uma:.0f} m")

    print(f"\n--- Path Loss Table (UMa NLOS) ---")
    print(f"  {'d [m]':>8}  {'PL [dB]':>8}  {'Status':>10}")
    for d in [100, 500, 800, 1000, 1500, 2000, 3000, 3680]:
        pl = pathloss_uma_nlos(d)
        st = "< L_max" if pl < Lmax else "= L_max" if abs(pl - Lmax) < 0.5 else "> L_max"
        print(f"  {d:>8}  {pl:>8.1f}  {st:>10}")

    print(f"\n--- Site Verification (R = 800 m, uniform) ---")
    sites = [
        ("Town Centre",       800),
        ("NH-48 North",       800),
        ("NH-48 South",       800),
        ("East / Dudhganga",  800),
        ("NE / MIDC Road",    800),
        ("West / Pimpalgaon", 800),
        ("SE / Outskirts",    800),
    ]
    ok = True
    for name, r in sites:
        pl = pathloss_uma_nlos(r)
        flag = "OK" if r < R_uma else "FAIL"
        if r >= R_uma:
            ok = False
        print(f"  {name:<20s}  R={r} m  PL={pl:.1f} dB  {flag}")

    print(f"\n  All sites within R_max? {'Yes' if ok else 'No'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
