"""
CFEM Limit State Design (LSD) & Reliability-Based Design (RBD) Engine
Author: Geotechnical Engineering Portfolio Project
Reference: Canadian Foundation Engineering Manual (CFEM) & CHBDC (CSA S6:19)
"""

import numpy as np


class CFEMLimitStateDesigner:
    """
    Implements Limit States Design (LSD) and Load and Resistance Factor Design (LRFD)
    for shallow and deep foundations based on CFEM Chapters 6, 7, and 8.
    """

    def __init__(self, consequence_level: str = "Typical", understanding_level: str = "Typical"):
        """
        Initializes consequence factors and resistance factors.

        Parameters:
        -----------
        consequence_level : str
            Structural consequence category: "High", "Typical", or "Low" (Table 6.4).
        understanding_level : str
            Degree of site and model understanding: "Low", "Typical", or "High" (Table 6.2).
        """
        # Consequence factor (Psi) based on CFEM Table 6.4 / CHBDC (CSA 2019)
        consequence_dict = {
            "High": 0.90,     # Essential lifeline structures, hospitals, major bridges
            "Typical": 1.00,  # Standard urban buildings and bridges
            "Low": 1.15       # Low risk to life, temporary or rarely visited structures
        }
        self.psi = consequence_dict.get(consequence_level, 1.00)
        self.consequence_level = consequence_level
        self.understanding_level = understanding_level

        # Geotechnical resistance factors (phi_gu, phi_gs) from CFEM Table 6.2
        phi_table = {
            "bearing_analysis": {"Low": 0.45, "Typical": 0.50, "High": 0.60},
            "settlement_analysis": {"Low": 0.70, "Typical": 0.80, "High": 0.90},
            "pile_static_analysis": {"Low": 0.35, "Typical": 0.40, "High": 0.45},
            "pile_static_test": {"Low": 0.50, "Typical": 0.60, "High": 0.70}
        }

        self.phi_gu_bearing = phi_table["bearing_analysis"][understanding_level]
        self.phi_gs_settlement = phi_table["settlement_analysis"][understanding_level]
        self.phi_gu_pile = phi_table["pile_static_analysis"][understanding_level]

    # =========================================================================
    # MODULE 1: SHALLOW FOUNDATION (ULS Bearing & SLS Consolidation Settlement)
    # =========================================================================
    def design_shallow_footing(
        self,
        V_factored: float,
        H_factored: float,
        V_service: float,
        soil_params: dict,
        H_clay: float = 4.5,
        delta_max: float = 0.025
    ) -> dict:
        """
        Designs square footing width B governed by both ULS and SLS criteria.

        Parameters:
        -----------
        V_factored : float
            Total factored vertical load at ULS (kN).
        H_factored : float
            Total factored horizontal load at ULS (kN).
        V_service : float
            Unfactored service vertical load at SLS (kN).
        soil_params : dict
            Dictionary containing soil mechanical and compressibility properties:
            'gamma', 'q_overburden', 'phi_deg', 'su', 'e0', 'Ccr', 'Cc', 'sigma_p_prime'.
        H_clay : float
            Compressible layer thickness below footing base (m).
        delta_max : float
            Maximum allowable settlement limit for SLS (m).

        Returns:
        --------
        dict: Summary of required dimensions for each limit state and governing mode.
        """
        gamma = soil_params['gamma']
        q_overburden = soil_params['q_overburden']
        phi_deg = soil_params['phi_deg']
        su = soil_params['su']

        # ---------------------------------------------------------
        # Load Inclination Factors (Meyerhof 1963 / CFEM Eq. 6.10)
        # ---------------------------------------------------------
        delta_f = np.degrees(np.arctan(H_factored / V_factored))  # Load inclination angle
        i_c = (1.0 - delta_f / 90.0)**2
        i_q = i_c
        i_gamma = (1.0 - delta_f / phi_deg)**2 if phi_deg > 0 else 0.0

        # ---------------------------------------------------------
        # 1. ULS Undrained Condition (phi = 0, su != 0)
        # ---------------------------------------------------------
        # Shape factors for square footing: sc = sq = 1.2; Nc = 5.14, Nq = 1.0
        qu_undrained = su * 5.14 * 1.2 * i_c + q_overburden * 1.0 * 1.2 * i_q
        factored_qu_undrained = self.psi * self.phi_gu_bearing * qu_undrained
        
        # Area = B^2 >= V_factored / (Psi * phi_gu * qu)
        B_uls_undrained = np.sqrt(V_factored / factored_qu_undrained)

        # ---------------------------------------------------------
        # 2. ULS Drained Condition (c' = 0, phi' > 0)
        # ---------------------------------------------------------
        phi_rad = np.radians(phi_deg)
        Nq = np.exp(np.pi * np.tan(phi_rad)) * (np.tan(np.pi / 4.0 + phi_rad / 2.0))**2
        Ngamma = 0.1054 * np.exp(0.1675 * phi_deg)
        sq, sgamma = 1.65, 0.6  # Shape factors for square footing

        # Iterative search for minimum width B satisfying LRFD inequality
        b_candidates = np.linspace(0.5, 10.0, 1000)
        qu_drained = (q_overburden * Nq * sq * i_q) + (0.5 * gamma * b_candidates * Ngamma * sgamma * i_gamma)
        factored_R_drained = self.psi * self.phi_gu_bearing * qu_drained * (b_candidates**2)
        valid_indices = np.where(factored_R_drained >= V_factored)[0]
        B_uls_drained = b_candidates[valid_indices[0]] if len(valid_indices) > 0 else np.nan

        # ---------------------------------------------------------
        # 3. SLS Settlement Requirement (Preconsolidated Clay Layer)
        # ---------------------------------------------------------
        e0 = soil_params['e0']
        Ccr = soil_params['Ccr']
        Cc = soil_params.get('Cc', 0.123)
        sigma_p_prime = soil_params.get('sigma_p_prime', 600.0)

        # In-situ vertical effective stress at layer mid-height
        sigma_0_prime = q_overburden + gamma * (H_clay / 2.0)

        # Back-calculate allowable stress increase (delta_sigma_prime) from consolidation equation
        # CFEM Eq. 6.17 (assuming final stress does not exceed preconsolidation pressure)
        delta_sigma_prime = sigma_0_prime * (10.0**((1.0 + e0) * delta_max / (H_clay * Ccr)) - 1.0)

        # Check if stress exceeds preconsolidation pressure (CFEM Eq. 6.18)
        if (sigma_0_prime + delta_sigma_prime) > sigma_p_prime:
            term1 = Ccr * np.log10(sigma_p_prime / sigma_0_prime)
            req_log_term2 = ((1.0 + e0) * delta_max / H_clay) - term1
            delta_sigma_prime = (sigma_p_prime * (10.0**(req_log_term2 / Cc))) - sigma_0_prime

        # Factored service resistance pressure: Psi * phi_gs * delta_sigma_prime
        factored_sls_capacity = self.psi * self.phi_gs_settlement * delta_sigma_prime
        B_sls = np.sqrt(V_service / factored_sls_capacity)

        # ---------------------------------------------------------
        # Governing Dimension Decision
        # ---------------------------------------------------------
        B_governing = max(B_uls_undrained, B_uls_drained, B_sls)
        governing_mode = "SLS (Settlement)" if B_governing == B_sls else "ULS (Bearing Capacity)"

        return {
            "B_uls_drained_m": round(float(B_uls_drained), 3),
            "B_uls_undrained_m": round(float(B_uls_undrained), 3),
            "B_sls_m": round(float(B_sls), 3),
            "Final_Designed_B_m": round(float(B_governing), 3),
            "Governing_Limit_State": governing_mode,
            "Effective_ULS_Resistance_Factor": round(self.psi * self.phi_gu_bearing, 3),
            "Effective_SLS_Resistance_Factor": round(self.psi * self.phi_gs_settlement, 3)
        }

    # =========================================================================
    # MODULE 2: DEEP FOUNDATION (Driven Pipe Pile Axial Resistance - Beta Method)
    # =========================================================================
    def design_driven_pile(
        self,
        factored_axial_load: float,
        soil_profile: list,
        pile_outer_dia: float = 0.356
    ) -> dict:
        """
        Determines the minimum required pile penetration length to resist factored axial load.
        Uses the beta-method (effective stress analysis) and end-bearing coefficient Nt (CFEM Eq. 6.21).

        Parameters:
        -----------
        factored_axial_load : float
            Total factored axial compressive demand (kN).
        soil_profile : list of dicts
            Stratigraphic layers, each specifying:
            'thickness' (m), 'gamma_eff' (kN/m3), 'beta' (shaft coeff), 'Nt' (toe factor).
        pile_outer_dia : float
            Outer diameter of the steel pipe pile (m).

        Returns:
        --------
        dict: Optimum pile length, factored capacities, and site understanding efficiency.
        """
        perimeter = np.pi * pile_outer_dia
        area_toe = np.pi * (pile_outer_dia**2) / 4.0

        depth_steps = np.linspace(1.0, 30.0, 300)
        optimum_length = None
        nominal_capacity_found = 0.0

        for L in depth_steps:
            current_depth = 0.0
            accumulated_shaft_R = 0.0
            sigma_v_prime = 0.0
            toe_layer = soil_profile[-1]

            for layer in soil_profile:
                t = layer['thickness']
                gamma = layer['gamma_eff']
                beta = layer['beta']
                
                z_top = current_depth
                z_bot = current_depth + t

                if L > z_top:
                    # Segment of pile inside this stratum
                    h_embed = min(L, z_bot) - z_top
                    sigma_mid = sigma_v_prime + gamma * (h_embed / 2.0)
                    
                    # Shaft resistance contribution: C * beta * sigma'_v * delta_z
                    accumulated_shaft_R += perimeter * beta * sigma_mid * h_embed
                    sigma_v_prime += gamma * h_embed
                    toe_layer = layer
                else:
                    break
                current_depth = z_bot

            # Toe resistance contribution: At * Nt * sigma'_v,toe
            toe_R = area_toe * toe_layer['Nt'] * sigma_v_prime
            R_ultimate = accumulated_shaft_R + toe_R

            # Check LRFD governing inequality: Psi * phi_gu * Ru >= Factored Load
            factored_R = self.psi * self.phi_gu_pile * R_ultimate
            if factored_R >= factored_axial_load:
                optimum_length = L
                nominal_capacity_found = R_ultimate
                break

        return {
            "Required_Pile_Length_m": round(float(optimum_length), 2) if optimum_length else None,
            "Nominal_Ultimate_Resistance_kN": round(float(nominal_capacity_found), 1),
            "Factored_Resistance_kN": round(float(self.psi * self.phi_gu_pile * nominal_capacity_found), 1),
            "Combined_Resistance_Factor": round(self.psi * self.phi_gu_pile, 3),
            "Site_Understanding_Level": self.understanding_level
        }


# =============================================================================
# VERIFICATION & CASE STUDY EXECUTION (CFEM Section 6.8 Benchmarking)
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("1. RUNNING SHALLOW FOOTING DESIGN BENCHMARK (Hospital - High Consequence)")
    print("=" * 70)

    # Input parameters from CFEM Section 6.8.1
    soil_props_shallow = {
        'gamma': 21.0,           # Soil unit weight (kN/m3)
        'q_overburden': 41.0,     # Effective overburden at foundation base (kPa)
        'phi_deg': 32.0,          # Effective internal friction angle (degrees)
        'su': 115.0,              # Undrained shear strength (kPa)
        'e0': 0.49,               # Initial void ratio
        'Ccr': 0.0135,            # Recompression index
        'Cc': 0.123,              # Compression index
        'sigma_p_prime': 600.0    # Preconsolidation pressure (kPa)
    }

    # High Consequence (Hospital) with Typical Site Understanding
    shallow_designer = CFEMLimitStateDesigner(consequence_level="High", understanding_level="Typical")
    shallow_result = shallow_designer.design_shallow_footing(
        V_factored=708.0,
        H_factored=100.0,
        V_service=530.0,
        soil_params=soil_props_shallow,
        H_clay=4.5,
        delta_max=0.025
    )

    for k, v in shallow_result.items():
        print(f" - {k}: {v}")

    print("\n" + "=" * 70)
    print("2. RUNNING DEEP FOUNDATION BENCHMARK (Overpass Bridge - Typical Consequence)")
    print("=" * 70)

    # Stratigraphy from CFEM Table 6.5
    # Comparing 3 boreholes (Low Understanding) vs 9 boreholes (High Understanding)
    stratigraphy_low = [
        {'thickness': 2.0, 'gamma_eff': 19.0, 'beta': 0.27, 'Nt': 27.0},
        {'thickness': 3.0, 'gamma_eff': 20.0, 'beta': 0.36, 'Nt': 45.0},
        {'thickness': 15.0, 'gamma_eff': 21.0, 'beta': 0.72, 'Nt': 81.0}
    ]

    stratigraphy_high = [
        {'thickness': 2.0, 'gamma_eff': 19.0, 'beta': 0.33, 'Nt': 32.0},
        {'thickness': 3.0, 'gamma_eff': 20.0, 'beta': 0.47, 'Nt': 61.0},
        {'thickness': 15.0, 'gamma_eff': 21.0, 'beta': 0.86, 'Nt': 100.0}
    ]

    # Analysis with Low Understanding (phi_gu = 0.35)
    pile_engine_low = CFEMLimitStateDesigner(consequence_level="Typical", understanding_level="Low")
    res_low = pile_engine_low.design_driven_pile(factored_axial_load=1100.0, soil_profile=stratigraphy_low)

    # Analysis with High Understanding (phi_gu = 0.45)
    pile_engine_high = CFEMLimitStateDesigner(consequence_level="Typical", understanding_level="High")
    res_high = pile_engine_high.design_driven_pile(factored_axial_load=1100.0, soil_profile=stratigraphy_high)

    print("Result with Low Understanding (3 Boreholes):")
    print(f" -> Required Length: {res_low['Required_Pile_Length_m']} m | Combined Factor: {res_low['Combined_Resistance_Factor']}")

    print("\nResult with High Understanding (9 Boreholes - Value Engineering):")
    print(f" -> Required Length: {res_high['Required_Pile_Length_m']} m | Combined Factor: {res_high['Combined_Resistance_Factor']}")

    length_saved = res_low['Required_Pile_Length_m'] - res_high['Required_Pile_Length_m']
    print(f"\n[Value Engineering Insight]: Additional site investigation saved {length_saved:.2f} m of pile length per unit!")
    print("=" * 70)
