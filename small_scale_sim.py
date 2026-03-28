#!/usr/bin/env python3
"""
4G LTE Radio Network Planning — Part 2: Small Scale Effects Simulation
======================================================================
Reference: 3GPP TR 38.901 V18.0.0 (2024-05), Section 7.5 (Steps 1-11)

This script simulates the Clustered Delay Line (CDL) model for:
  - Scenario: UMa (Urban Macrocell) NLOS
  - Frequency: 1.8 GHz
  - Antenna Configuration: 2x8 MIMO Channel Matrix (Step 11)

Author: Aditya Kinjawadekar
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================================
# 1. SIMULATION PARAMETERS (Step 1)
# ============================================================================

f_c = 1.8e9        # 1.8 GHz
c = 3.0e8          # light speed [m/s]
lambda_c = c / f_c

# Antenna Array Configuration
# BS: 4x2 cross-pol ULA (8 elements)
n_bs_beams = 4
n_bs_pol = 2
n_bs = n_bs_beams * n_bs_pol  # 8 elements

# UT: 2-element array (1x2)
n_ut = 2

# Number of Clusters and Rays (UMa NLOS Table 7.5-6)
n_clusters = 20
m_rays = 20

# Large Scale Parameters (LSPs) for UMa NLOS @ 1.8 GHz [RMS]
# Taken from 3GPP TR 38.901 Table 7.5-6
# mu_log_X, sigma_log_X
log_ds = -6.63    # Delay Spread [log10(s)]
std_log_ds = 0.32
log_asd = 1.0     # Azimuth Spread Departure [log10(deg)]
log_asa = 1.5     # Azimuth Spread Arrival [log10(deg)]
log_zsd = 0.5     # Zenith Spread Departure [log10(deg)]
log_zsa = 0.8     # Zenith Spread Arrival [log10(deg)]
sf_std = 6.0      # Shadow Fading [dB]

# ============================================================================
# 3. LINK BUDGET & ENVIRONMENT (Step 2 & 3)
# ============================================================================

# Position Setup
bs_pos = np.array([0, 0, 25])  # h_BS = 25m
ut_pos = np.array([500, 300, 1.5])  # h_UT = 1.5m at 583m distance

d_2d = np.linalg.norm(ut_pos[:2] - bs_pos[:2])
d_3d = np.linalg.norm(ut_pos - bs_pos)

# ============================================================================
# 4. GENERATE DELAYS & POWERS (Step 4 & 5)
# ============================================================================

def generate_delays_and_powers():
    # RMS DS [sec]
    ds = 10**log_ds
    
    # Generate delays (Step 4) - simplified exponential PDF
    tau = np.random.uniform(0, 5, n_clusters) * ds
    tau = np.sort(tau - np.min(tau))
    
    # Generate powers (Step 5) - P' = exp(-tau/DS)
    # With log-normal variation
    z_n = np.random.normal(0, 3, n_clusters) # Cluster power variation [dB]
    p_prime = np.exp(-tau / ds) * (10**(z_n/10.0))
    p = p_prime / np.sum(p_prime) # Normalized
    
    return tau, p

# ============================================================================
# 5. GENERATE ANGLES (Step 6)
# ============================================================================

def generate_angles():
    # ASD, ASA from LSPs
    asd = 10**log_asd
    asa = 10**log_asa
    zsd = 10**log_zsd
    zsa = 10**log_zsa
    
    # Step 6: Generate AoD, AoA, ZoD, ZoA for each cluster
    # Using Laplacian distribution offsets as per 3GPP
    aod = np.random.normal(0, asd, n_clusters)
    aoa = np.random.normal(0, asa, n_clusters)
    zod = 90 + np.random.normal(0, zsd, n_clusters) # Elevation relative to vertical
    zoa = 90 + np.random.normal(0, zsa, n_clusters)
    
    return aod, aoa, zod, zoa

# ============================================================================
# 6. CHANNEL COEFFICIENTS SYNTHESIS (Step 11 - BONUS)
# ============================================================================

def generate_channel_matrix(tau, p, aod, aoa, zod, zoa):
    """
    Constructs the H_u,s,n(t) matrix (Step 11) for UT u, BS s at time t=0.
    
    H[u][s] = sum_n sqrt(p_n/M) * sum_m exp(j * (phi_n,m + antenna_phases))
    """
    H = np.zeros((n_ut, n_bs), dtype=complex)
    
    # Phase initialization
    phi = np.random.uniform(0, 2*np.pi, (n_clusters, m_rays))
    
    # Antenna coordinates (simplified spacing d = lambda/2)
    d = lambda_c / 2
    bs_ant_x = np.linspace(0, (n_bs_beams-1)*d, n_bs_beams)
    ut_ant_x = np.linspace(0, (n_ut-1)*d, n_ut)
    
    # Polarization offset (Simplification: vertical)
    # sum across clusters and rays
    for n in range(n_clusters):
        sqrt_pn = np.sqrt(p[n])
        
        # Ray offsets within cluster (3GPP Table 7.5-3)
        # Small offsets around the cluster central angle
        ray_offsets = np.random.normal(0, 2, m_rays) # Degrees
        
        for m in range(m_rays):
            alpha_d = (aod[n] + ray_offsets[m]) * (np.pi/180)
            alpha_a = (aoa[n] + ray_offsets[m]) * (np.pi/180)
            
            # Phase terms for UT antennas
            ut_phase = np.exp(1j * 2 * np.pi * ut_ant_x * np.sin(alpha_a) / lambda_c)
            
            # Phase terms for BS antennas (accounting for pol)
            bs_phase = np.exp(1j * 2 * np.pi * bs_ant_x * np.sin(alpha_d) / lambda_c)
            
            # Combine into H matrix (Step 11 final sum)
            # H[u,s] += sqrt(p_n/M) * exp(j*phi_n,m) * ut_array_resp * bs_array_resp
            for u in range(n_ut):
                # We replicate cross-pol by adding second set of elements with 90deg phase
                for s_beam in range(n_bs_beams):
                    # Pol 1 (Vertical)
                    idx1 = s_beam * 2
                    H[u, idx1] += (sqrt_pn / np.sqrt(m_rays)) * np.exp(1j * phi[n,m]) * ut_phase[u] * bs_phase[s_beam]
                    
                    # Pol 2 (Horizontal - simplified phase shift)
                    idx2 = s_beam * 2 + 1
                    H[u, idx2] += (sqrt_pn / np.sqrt(m_rays)) * np.exp(1j * (phi[n,m] + np.pi/2)) * ut_phase[u] * bs_phase[s_beam]

    return H

# ============================================================================
# RESULTS & PLOTTING
# ============================================================================

def main():
    print("Evaluating 3GPP 38.901 Fast Fading steps for UMa NLOS...")
    
    tau, p = generate_delays_and_powers()
    aod, aoa, zod, zoa = generate_angles()
    H = generate_channel_matrix(tau, p, aod, aoa, zod, zoa)
    
    # Print results
    print(f"\nRMS Delay Spread: {np.sqrt(np.sum(p * tau**2) - (np.sum(p * tau))**2)*1e9:.2f} ns")
    print(f"Channel Matrix H (2x8) Mag [dB]:")
    h_db = 20 * np.log10(np.abs(H))
    print(np.round(h_db, 1))
    
    # Save Power Delay Profile Plot
    plt.figure(figsize=(10, 6))
    plt.stem(tau * 1e9, 10 * np.log10(p), basefmt=" ")
    plt.xlabel('Delay [ns]')
    plt.ylabel('Power [dB]')
    plt.title('UMa NLOS Power Delay Profile (Step 4 & 5)')
    plt.grid(True)
    plt.savefig('pdp_plot.png')
    
    # Save Matrix Heatmap
    plt.figure(figsize=(8, 4))
    plt.imshow(h_db, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude [dB]')
    plt.xlabel('BS Antenna Index (4x2 Cross-Pol)')
    plt.ylabel('UT Antenna Index')
    plt.title('Channel Matrix Magnitude Heatmap (Step 11 Bonus)')
    plt.savefig('h_matrix_heatmap.png')
    
    print("\nPlots saved: pdp_plot.png, h_matrix_heatmap.png")

if __name__ == "__main__":
    main()
